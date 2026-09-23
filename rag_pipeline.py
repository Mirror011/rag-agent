"""RAG 核心流程：解析、分块、建库、检索、生成"""
import os
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import config


def load_pdf(pdf_path):
    """解析 PDF，返回全文文字"""
    print(f"正在解析 PDF:{pdf_path} ...")
    doc = pymupdf.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    print(f"解析完成!共 {len(full_text)} 个字符")
    return full_text


def split_chunks(full_text):
    """把全文切分成文本块"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = text_splitter.split_text(full_text)
    documents = [Document(page_content=chunk) for chunk in chunks]
    print(f"切分完成!共 {len(documents)} 个文本块")
    return documents


def get_embeddings():
    """加载嵌入模型"""
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


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
        full_text = load_pdf(config.PDF_PATH)
        chunks = split_chunks(full_text)
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


def ask(vectorstore, llm, question):
    """检索 + 生成答案"""
    retrieved_docs = vectorstore.similarity_search(question, k=config.TOP_K)
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