import os
import pymupdf
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

# 向量库存放目录（持久化到硬盘）
PERSIST_DIR = "chroma_db"


def load_pdf(pdf_path):
    """第1步:读取 PDF"""
    print(f"正在解析 PDF:{pdf_path} ...")
    doc = pymupdf.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    print(f"解析完成!共 {len(full_text)} 个字符")
    return full_text


def split_chunks(full_text, chunk_size=1000, chunk_overlap=100):
    """第2步:分块"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(full_text)
    documents = [Document(page_content=chunk) for chunk in chunks]
    print(f"切分完成!共 {len(documents)} 个文本块")
    return documents


def get_embeddings():
    """加载嵌入模型（抽出来单独一个函数，加载库和建库都要用）"""
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


def get_vectorstore(pdf_path):
    """第3步:获取向量库——有持久化的就直接加载，没有才新建"""
    embeddings = get_embeddings()

    # 判断硬盘上是否已经建过库
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        print("检测到已存在的向量库，直接加载（秒级）...")
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings
        )
    else:
        print("未检测到向量库，开始新建（首次较慢）...")
        full_text = load_pdf(pdf_path)
        chunks = split_chunks(full_text)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIR  # 关键：指定存盘目录
        )
        print("向量库已构建并保存到硬盘!")
    return vectorstore


def ask(vectorstore, llm, question):
    """第4步:检索 + 生成答案"""
    retrieved_docs = vectorstore.similarity_search(question, k=5)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    prompt = ChatPromptTemplate.from_template(
        """你是一个财报分析助手。请根据下面提供的财报内容回答问题。
请确保回答完全基于这些内容,如果内容里找不到答案,就直接说"根据提供的财报内容,无法回答此问题"。

财报内容:
{context}

问题:{question}

回答:"""
    )
    chain = prompt | llm
    answer = chain.invoke({"context": context, "question": question})
    return answer.content


# ============ 主程序 ============
if __name__ == "__main__":
    pdf_path = "data/tsla-20251231.pdf"

    # 准备向量库（首次建库，之后加载）
    vectorstore = get_vectorstore(pdf_path)

    # 准备大模型（只需初始化一次，循环里复用）
    llm = ChatOpenAI(
        model="deepseek-chat",
        temperature=0,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com"
    )

    # 连续对话循环
    print("\n" + "=" * 60)
    print("财报问答助手已就绪!输入你的问题，输入 quit 退出。")
    print("=" * 60)

    while True:
        question = input("\n请输入问题：").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("再见!")
            break
        if not question:
            continue
        print("正在思考...")
        answer = ask(vectorstore, llm, question)
        print(f"\n回答:{answer}")