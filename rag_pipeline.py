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

def parse_sections(pages):
    """
    识别 10-K 的章节边界。
    依据：正文章节标题为全大写（如 ITEM 7. MANAGEMENT'S...），
    目录页为首字母大写（Item 7.），据此过滤目录与正文中的法条引用。
    返回 [(字符起始位置, ITEM编号, 标题), ...]
    """
    full_text = ""
    page_offsets = []  # (该页起始字符位置, 页码)
    for pno, text in pages:
        page_offsets.append((len(full_text), pno))
        full_text += text

    # 必须全大写 + ITEM + 编号 + 点，且后面跟着标题文字（排除 "Item 408(a)" 这类引用）
    pattern = re.compile(r"^(ITEM\s+(\d+[A-C]?))\.\s+([A-Z][A-Z\s,’'\-&\.\(\)\[\]/]{4,})$", re.MULTILINE)

    sections = []
    for m in pattern.finditer(full_text):
        item_no = "ITEM " + m.group(2)
        title = m.group(3).strip()
        sections.append((m.start(), item_no, title))

    print(f"识别到 {len(sections)} 个正文章节")
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
    """解析 PDF，返回 [(页码, 该页文字), ...]"""
    print(f"正在解析 PDF:{pdf_path} ...")
    doc = pymupdf.open(pdf_path)
    pages = []
    for i, page in enumerate(doc, start=1):
        pages.append((i, page.get_text()))
    doc.close()
    total = sum(len(t) for _, t in pages)
    print(f"解析完成!共 {len(pages)} 页，{total} 个字符")
    return pages


def split_chunks(pages):
    """切分文本块，并为每块附加元数据"""
    sections, full_text, page_offsets = parse_sections(pages)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = text_splitter.split_text(full_text)

    documents = []
    cursor = 0
    for chunk in chunks:
        # 定位该块在全文中的位置
        pos = full_text.find(chunk, cursor)
        if pos == -1:
            pos = cursor
        cursor = pos + len(chunk) - config.CHUNK_OVERLAP

        section, title, page = locate(pos, sections, page_offsets)
        documents.append(Document(
            page_content=chunk,
            metadata={
                "company": config.DOC_COMPANY,
                "fiscal_year": config.DOC_FISCAL_YEAR,
                "doc_type": config.DOC_TYPE,
                "section": section,
                "section_title": title[:80],
                "section_type": config.SECTION_TYPE_MAP.get(section, "other"),
                "page": page,
            }
        ))

    print(f"切分完成!共 {len(documents)} 个文本块（已附元数据）")
    return documents


def get_embeddings():
    """加载嵌入模型"""
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

# 全局缓存重排模型，避免每次提问都重新加载
_reranker = None

def get_reranker():
    """加载重排模型（CrossEncoder），只加载一次"""
    global _reranker
    if _reranker is None:
        print("正在加载重排模型 bge-reranker（首次会下载）...")
        _reranker = CrossEncoder(config.RERANK_MODEL, max_length=512)
    return _reranker

def get_vectorstore():
    """获取向量库：有持久化则加载，否则新建"""
    embeddings = get_embeddings()

    if os.path.exists(config.PERSIST_DIR) and os.listdir(config.PERSIST_DIR):
        print("检测到已存在的向量库，直接加载...")
        vectorstore = Chroma(
            persist_directory=config.PERSIST_DIR,
            embedding_function=embeddings
        )
    else:
        print("未检测到向量库，开始新建...")
        pages = load_pdf(config.PDF_PATH)
        chunks = split_chunks(pages)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=config.PERSIST_DIR
        )
        print("向量库已构建并保存!")
    return vectorstore


def get_llm():
    """初始化大模型"""
    return ChatOpenAI(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=config.LLM_BASE_URL
    )


def detect_section_type(question):
    """根据问题关键词推断应该检索哪类章节，判断不出则返回 None（不过滤）"""
    q = question.lower()
    for stype, keywords in config.INTENT_KEYWORDS.items():
        if any(kw.lower() in q for kw in keywords):
            return stype
    return None

# 常见财务科目关键词 → 数据库查询词
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


def lookup_structured(question):
    """从问题中识别财务科目，查结构化库，返回精确数据"""
    q = question.lower()
    hits = []
    for kw, dbterm in FIN_KEYWORDS.items():
        if kw.lower() in q:
            hits.append(dbterm)
    if not hits:
        return ""

    rows = []
    for term in dict.fromkeys(hits):      # 去重保序
        rows.extend(table_db.query_trend(term, limit=12))
    return table_db.format_results(rows[:40])

def ask(vectorstore, llm, question):
    """检索 + 重排（含章节软加权）+ 生成"""
    stype = detect_section_type(question)
    if stype:
        print(f"[章节偏好] {stype} 类章节将在重排时加权（不排除其他章节）")

    if config.USE_RERANK:
        # 关键改动：不再传 filter，全范围召回
        candidates = vectorstore.similarity_search(question, k=config.RETRIEVE_TOP_K)
        print(f"[召回候选章节] {[d.metadata.get('section') for d in candidates]}")
        reranker = get_reranker()
        pairs = [[question, doc.page_content] for doc in candidates]
        scores = reranker.predict(pairs)

        # 软加权：相关章节的候选加分，但不剔除其他章节
        adjusted = []
        for score, doc in zip(scores, candidates):
            if stype and doc.metadata.get("section_type") == stype:
                score = score + config.SECTION_BOOST
            adjusted.append((score, doc))

        adjusted.sort(key=lambda x: x[0], reverse=True)
        retrieved_docs = [doc for _, doc in adjusted[:config.RERANK_TOP_K]]
    else:
        retrieved_docs = vectorstore.similarity_search(question, k=config.RERANK_TOP_K)

    # 组装上下文，带出处
    parts = []
    for d in retrieved_docs:
        src = f"[来源：{d.metadata.get('section','?')}，第 {d.metadata.get('page','?')} 页]"
        parts.append(f"{src}\n{d.page_content}")
    context = "\n\n".join(parts)

    # 新增：结构化财务数据（精确数值，优先级高于文本块）
    structured = lookup_structured(question)
    if structured:
        print("[结构化查询] 命中财务科目，已注入精确数据")
        context = structured + "\n\n" + context

    # 打印实际用了哪些章节，方便验证
    used = [d.metadata.get("section", "?") for d in retrieved_docs]
    print(f"[实际使用章节] {used}")

    prompt = ChatPromptTemplate.from_template(
        """你是一个财报分析助手。请根据下面提供的财报内容回答问题。

如果内容中包含"[结构化财务数据]"部分，请优先使用其中的数字，因为它们是从财报表格中
精确提取的，年份与科目对应关系可靠；文本部分仅作为补充说明。

请确保回答完全基于这些内容，并在答案末尾标注依据的来源（章节与页码）。
如果问题需要计算（如增长率、占比、差额），可以基于上述数字进行计算并写出计算过程。
如果内容里找不到答案，就直接说"根据提供的财报内容，无法回答此问题"。

财报内容:
{context}

问题:{question}

回答:"""
    )
    chain = prompt | llm
    answer = chain.invoke({"context": context, "question": question})
    return answer.content

