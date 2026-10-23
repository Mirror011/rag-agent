"""项目配置：集中管理路径、模型、参数"""
import os
# 设置 huggingface 国内镜像（必须放在最前面，导入相关库之前生效）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # 顺便关掉那个警告

# 数据与存储路径
PDF_PATH = "data/tsla-20251231.pdf"
PERSIST_DIR = "chroma_db"

# 分块参数
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# 嵌入模型
EMBEDDING_MODEL = "BAAI/bge-m3"

# 检索参数
TOP_K = 5

# 大模型配置
LLM_MODEL = "deepseek-chat"
LLM_BASE_URL = "https://api.deepseek.com"
LLM_TEMPERATURE = 0

# 重排(Rerank)配置
USE_RERANK = True                              # 是否启用重排（方便对比开/关效果）
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"       # 重排模型（多语言，和bge-m3同门）
RETRIEVE_TOP_K = 20                            # 第1段：向量检索粗筛的候选数量
RERANK_TOP_K = 5                               # 第2段：重排后最终保留的数量

# ===== 元数据配置 =====
# 文档基本信息（后续支持多公司时改为按文件配置）
DOC_COMPANY = "Tesla"
DOC_FISCAL_YEAR = 2025
DOC_TYPE = "10-K"

# SEC 法定章节 → 语义分类映射
# 10-K 的 Item 编号由 SEC 强制规定，所有在美上市公司一致，故此表可跨公司复用
SECTION_TYPE_MAP = {
    "ITEM 1": "business", "ITEM 1A": "risk", "ITEM 1B": "business",
    "ITEM 1C": "business", "ITEM 2": "business", "ITEM 3": "legal",
    "ITEM 4": "business", "ITEM 5": "market", "ITEM 6": "financial",
    "ITEM 7": "financial", "ITEM 7A": "risk", "ITEM 8": "financial",
    "ITEM 9": "governance", "ITEM 9A": "governance", "ITEM 9B": "governance",
    "ITEM 9C": "governance", "ITEM 10": "governance", "ITEM 11": "governance",
    "ITEM 12": "governance", "ITEM 13": "governance", "ITEM 14": "governance",
    "ITEM 15": "appendix", "ITEM 16": "appendix",
}

# 问题关键词 → 章节类型（用于检索过滤）
INTENT_KEYWORDS = {
    "risk": ["risk", "风险", "threat", "uncertainty"],
    "financial": ["revenue", "income", "profit", "cost", "margin", "cash",
                  "营收", "收入", "利润", "成本", "毛利", "现金"],
    "business": ["product", "business", "operation", "factory", "产品", "业务", "工厂"],
    "governance": ["director", "executive", "compensation", "董事", "高管", "薪酬"],
}