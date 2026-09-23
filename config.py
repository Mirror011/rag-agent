"""项目配置：集中管理路径、模型、参数"""

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