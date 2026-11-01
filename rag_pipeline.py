"""RAG 核心流程：解析、分块、建库、检索、生成"""
import re
import os
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from sentence_transformers import CrossEncoder
import table_db

import config


# ============ 文档解析 ============

def parse_sections(pages):
    """
    识别 10-K 的章节边界。
    返回 [(字符起始位置, ITEM编号, 标题), ...]
    """
    full_text = ""
    page_offsets = []  # (该页起始字符位置, 页码)
    for pno, text in pages:
        page_offsets.append((len(full_text), pno))
        full_text += text

    # 兼容两种排版：
    #   特斯拉 ITEM 7. MANAGEMENT'S...（全大写）
    #   苹果   Item 7.\xa0\xa0Management's...（首字母大写 + 不间断空格）
    norm_text = full_text.replace("\xa0", " ")
    pattern = re.compile(
        r"^(ITEM|Item)\s+(\d+[A-C]?)\.\s+([A-Za-z][A-Za-z\s,’'\-&\.\(\)\[\]/]{4,})$",
        re.MULTILINE)

    sections = []
    for m in pattern.finditer(norm_text):
        item_no = "ITEM " + m.group(2).upper()
        title = m.group(3).strip()
        sections.append((m.start(), item_no, title))

    print(f"    识别到 {len(sections)} 个正文章节")
    return sections, full_text, page_offsets

def locate(pos, sections, page_offsets):
    """给定字符位置，返回它属于哪个章节、哪一页"""
    section, title = "UNKNOWN", ""
    for start, item_no, t in sections:
        if start <= pos:
            section, title = item_no, t
        else:
            break
    page = 1
    for offset, pno in page_offsets:
        if offset <= pos:
            page = pno
        else:
            break
    return section, title, page


def load_pdf(pdf_path):
    """解析 PDF，返回 ([(页码, 文字), ...], doc 对象)
    注意：doc 不在此关闭，需由调用方 close，供表格提取使用。"""
    print(f"    正在解析 {pdf_path} ...")
    doc = pymupdf.open(pdf_path)
    pages = [(i, page.get_text()) for i, page in enumerate(doc, start=1)]
    total = sum(len(t) for _, t in pages)
    print(f"    共 {len(pages)} 页，{total} 个字符")
    return pages, doc


# ============ 表格块提取 ============

def _clean_row(row):
    """去掉空列和纯符号列"""
    return [c.strip() for c in row
            if c and c.strip() not in ("", "$", "%", "(", ")")]


def _detect_years(page_text):
    """识别页面上的年份序列（财报通常列三年）"""
    years, seen = [], set()
    for y in re.findall(r"\b(20[0-3]\d)\b", page_text):
        if y not in seen:
            seen.add(y)
            years.append(y)
    return years[:3]


def extract_tables(doc):
    """提取表格并转为保留行列结构的 Markdown 块，返回 [(页码, 文本), ...]"""
    table_docs = []
    for pno in range(len(doc)):
        page = doc[pno]
        try:
            tabs = page.find_tables()
        except Exception:
            continue
        if not tabs.tables:
            continue

        years = _detect_years(page.get_text())
        for tab in tabs.tables:
            rows = [_clean_row(r) for r in tab.extract()]
            rows = [r for r in rows if len(r) >= 2]
            if len(rows) < 2:
                continue

            header = f"[表格] 第 {pno + 1} 页"
            if years:
                header += f"　列对应年份（从左到右）：{' / '.join(years)}"
            lines = ["| " + " | ".join(r) + " |" for r in rows]
            table_docs.append((pno + 1, header + "\n" + "\n".join(lines)))

    print(f"    提取到 {len(table_docs)} 张结构化表格")
    return table_docs


# ============ 分块 ============

def split_chunks(pages, doc=None, cfg=None):
    """切分正文文本块，并补充保留行列结构的表格块"""
    cfg = cfg or {"company": config.DOC_COMPANY, "year": config.DOC_FISCAL_YEAR}
    sections, full_text, page_offsets = parse_sections(pages)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = text_splitter.split_text(full_text)

    def make_meta(section, title, page, ctype):
        return {
            "company": cfg["company"],
            "fiscal_year": cfg["year"],
            "doc_type": config.DOC_TYPE,
            "section": section,
            "section_title": title[:80],
            "section_type": config.SECTION_TYPE_MAP.get(section, "other"),
            "page": page,
            "content_type": ctype,
        }

    documents = []
    cursor = 0
    for chunk in chunks:
        pos = full_text.find(chunk, cursor)
        if pos == -1:
            pos = cursor
        cursor = pos + len(chunk) - config.CHUNK_OVERLAP
        section, title, page = locate(pos, sections, page_offsets)
        documents.append(Document(
            page_content=chunk,
            metadata=make_meta(section, title, page, "text")
        ))
    n_text = len(documents)

    # 补充结构化表格块
    if doc is not None:
        page_start = {pno: off for off, pno in page_offsets}
        for pno, content in extract_tables(doc):
            pos = page_start.get(pno, 0)
            section, title, _ = locate(pos, sections, page_offsets)
            documents.append(Document(
                page_content=content,
                metadata=make_meta(section, title, pno, "table")
            ))

    print(f"    切分完成：{n_text} 文本块 + {len(documents) - n_text} 表格块")
    return documents


# ============ 模型与向量库 ============

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


_reranker = None


def get_reranker():
    """加载重排模型（CrossEncoder），只加载一次"""
    global _reranker
    if _reranker is None:
        print("正在加载重排模型 bge-reranker ...")
        _reranker = CrossEncoder(config.RERANK_MODEL, max_length=512)
    return _reranker


def get_vectorstore():
    """获取向量库：有持久化则加载，否则遍历所有公司财报新建"""
    embeddings = get_embeddings()

    if os.path.exists(config.PERSIST_DIR) and os.listdir(config.PERSIST_DIR):
        print("检测到已存在的向量库，直接加载...")
        return Chroma(
            persist_directory=config.PERSIST_DIR,
            embedding_function=embeddings
        )

    print("未检测到向量库，开始新建...")
    chunks = []
    for cfg in config.DOCUMENTS:
        if not os.path.exists(cfg["path"]):
            print(f"  跳过（文件不存在）：{cfg['path']}")
            continue
        print(f"  处理 {cfg['company']} ...")
        pages, doc = load_pdf(cfg["path"])
        chunks.extend(split_chunks(pages, doc, cfg))
        doc.close()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=config.PERSIST_DIR
    )
    print(f"向量库已构建并保存！共 {len(chunks)} 块")
    return vectorstore


def get_llm():
    return ChatOpenAI(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=config.LLM_BASE_URL
    )


# ============ 查询理解 ============

def detect_section_type(question):
    """根据问题关键词推断应优先检索哪类章节"""
    q = question.lower()
    for stype, keywords in config.INTENT_KEYWORDS.items():
        if any(kw.lower() in q for kw in keywords):
            return stype
    return None


FIN_KEYWORDS = {
    "营收": "total revenues", "总营收": "total revenues", "revenue": "revenues",
    "净利润": "net income", "net income": "net income",
    "毛利": "gross profit", "gross profit": "gross profit",
    "成本": "cost of revenues", "cost": "cost of revenues",
    "研发": "research and development", "r&d": "research and development",
    "汽车": "automotive", "automotive": "automotive",
    "能源": "energy", "energy": "energy",
    "服务": "services", "services": "services",
    "现金": "cash", "cash": "cash",
}

COMPANY_ALIASES = {
    "Tesla": ("特斯拉", "tesla", "tsla"),
    "Apple": ("苹果", "apple", "aapl"),
}


def detect_companies(question):
    """识别问题涉及哪些公司；未提及则返回 [None] 表示不限定"""
    q = question.lower()
    found = [c for c, alias in COMPANY_ALIASES.items()
             if any(a in q for a in alias)]
    return found or [None]


def lookup_structured(question):
    """识别财务科目与公司，从结构化库取精确数值"""
    q = question.lower()
    hits = [db for kw, db in FIN_KEYWORDS.items() if kw.lower() in q]
    if not hits:
        return ""

    companies = detect_companies(question)

    rows = []
    for comp in companies:
        for term in dict.fromkeys(hits):
            rows.extend(table_db.query_trend(term, company=comp, limit=12))
    out = table_db.format_results(rows[:60])

    # 构成分析类问题，附加科目层级
    if any(k in q for k in ("构成", "拖累", "占比", "子项", "分项", "哪个",
                            "breakdown", "contribut", "which")):
        for term in dict.fromkeys(hits):
            children = table_db.query_children(f"Total {term}")
            if children:
                out += "\n\n[科目构成明细]\n"
                seen = set()
                for item, year, value, page in children[:20]:
                    if (item, year) in seen:
                        continue
                    seen.add((item, year))
                    out += f"{item} | {year}年 | {value:,.0f} | 第{page}页\n"
                break
    return out


# ============ 问答 ============

def ask(vectorstore, llm, question):
    """检索 + 重排（章节软加权 + 公司软加权）+ 生成"""
    stype = detect_section_type(question)
    companies = [c for c in detect_companies(question) if c]
    if stype:
        print(f"[章节偏好] {stype}")
    if companies:
        print(f"[公司偏好] {companies}")

    if config.USE_RERANK:
        candidates = vectorstore.similarity_search(
            question, k=config.RETRIEVE_TOP_K)
        reranker = get_reranker()
        pairs = [[question, d.page_content] for d in candidates]
        scores = reranker.predict(pairs)

        adjusted = []
        for score, d in zip(scores, candidates):
            if stype and d.metadata.get("section_type") == stype:
                score += config.SECTION_BOOST
            # 跨公司问题：被提及的公司加权，保证双方都有机会入选
            if companies and d.metadata.get("company") in companies:
                score += config.SECTION_BOOST
            adjusted.append((score, d))

        adjusted.sort(key=lambda x: x[0], reverse=True)
        retrieved_docs = [d for _, d in adjusted[:config.RERANK_TOP_K]]
    else:
        retrieved_docs = vectorstore.similarity_search(
            question, k=config.RERANK_TOP_K)

    parts = []
    for d in retrieved_docs:
        src = (f"[来源：{d.metadata.get('company','?')} "
               f"{d.metadata.get('section','?')}，第 {d.metadata.get('page','?')} 页]")
        parts.append(f"{src}\n{d.page_content}")
    context = "\n\n".join(parts)

    structured = lookup_structured(question)
    if structured:
        print("[结构化查询] 命中财务科目，已注入精确数据")
        context = structured + "\n\n" + context

    used = [f"{d.metadata.get('company','?')}/{d.metadata.get('section','?')}"
            for d in retrieved_docs]
    print(f"[实际使用] {used}")

    prompt = ChatPromptTemplate.from_template(
        """你是一个财报分析助手。请根据下面提供的财报内容回答问题。

如果内容中包含"[结构化财务数据]"部分，请优先使用其中的数字，因为它们是从财报表格中
精确提取的，公司、年份与科目的对应关系可靠；文本部分仅作为补充说明。
涉及多家公司时，务必注意区分数据所属的公司，不要混用。

请确保回答完全基于这些内容，并在答案末尾标注依据的来源（公司、章节与页码）。
如果问题需要计算（如增长率、占比、差额），可以基于上述数字进行计算并写出计算过程。
如果内容里找不到答案，就直接说"根据提供的财报内容，无法回答此问题"。

财报内容:
{context}

问题:{question}

回答:"""
    )
    chain = prompt | llm
    return chain.invoke({"context": context, "question": question}).content
