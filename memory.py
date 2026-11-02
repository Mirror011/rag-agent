"""长期记忆：向量检索 + SQLite 双写

设计说明
--------
两种存储各司其职：
    Chroma（向量库）：按语义检索与当前问题相关的记忆，解决"记忆变多后无法全量注入"的问题
    SQLite        ：保存完整记录与元数据，负责去重、按时间排序、列出全部、清空

复用项目已有的 bge-m3 嵌入模型，与财报文档检索共用一套向量化能力，
但使用独立的 collection，两类数据互不干扰。

三类记忆：
    profile    用户画像（关注的公司、指标）
    preference 回答偏好（简洁、表格、语言等）
    finding    历史分析结论（带数字的关键发现）
"""
import os
import sqlite3
import datetime
import threading

from langchain_core.documents import Document
from langchain_chroma import Chroma

import config

DB_PATH = getattr(config, "MEMORY_DB_PATH", "memory.db")
VEC_DIR = getattr(config, "MEMORY_VEC_DIR", "memory_vec")
COLLECTION = "long_term_memory"

MAX_PER_USER = 200          # 每个用户最多保留的记忆条数
TOP_K = 5                   # 每次检索注入的记忆条数
MIN_SCORE = 0.30            # 相关度阈值，低于此值视为不相关而丢弃

LABELS = {"profile": "用户关注", "preference": "回答偏好", "finding": "历史结论"}

_VEC = None
_LOCK = threading.Lock()


# ============================================================
# 底层存储
# ============================================================

def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    TEXT,
            kind       TEXT,
            content    TEXT,
            created_at TEXT
        )
    """)
    return conn


def _vec():
    """记忆专用向量库，懒加载并加锁（与 agent 的并发工具调用共存）"""
    global _VEC
    if _VEC is None:
        with _LOCK:
            if _VEC is None:
                import rag_pipeline          # 延迟导入，避免循环依赖
                _VEC = Chroma(
                    collection_name=COLLECTION,
                    persist_directory=VEC_DIR,
                    embedding_function=rag_pipeline.get_embeddings(),
                )
    return _VEC


# ============================================================
# 写入
# ============================================================

def add_many(user_id, items):
    """写入多条记忆（SQLite 去重 + 同步写入向量库），返回新增条数"""
    new_docs, new_ids, added = [], [], 0

    with _conn() as c:
        for it in items:
            kind = (it.get("kind") or "").strip()
            content = (it.get("content") or "").strip()
            if kind not in LABELS or not content:
                continue
            if c.execute("SELECT 1 FROM memories WHERE user_id=? AND content=?",
                         (user_id, content)).fetchone():
                continue                     # 内容完全相同，跳过

            ts = datetime.datetime.now().isoformat(timespec="seconds")
            cur = c.execute(
                "INSERT INTO memories (user_id, kind, content, created_at) VALUES (?,?,?,?)",
                (user_id, kind, content, ts))
            new_ids.append(str(cur.lastrowid))
            new_docs.append(Document(
                page_content=content,
                metadata={"user_id": user_id, "kind": kind,
                          "created_at": ts, "mem_id": cur.lastrowid},
            ))
            added += 1

    if new_docs:
        try:
            _vec().add_documents(new_docs, ids=new_ids)
        except Exception as e:
            print(f"  [记忆] 向量写入失败（SQLite 已保存）：{e}")

    _prune(user_id)
    return added


def _prune(user_id):
    """超出上限时删除最旧的记忆，向量库同步删除"""
    with _conn() as c:
        stale = c.execute("""
            SELECT id FROM memories WHERE user_id = ? AND id NOT IN (
                SELECT id FROM memories WHERE user_id = ?
                ORDER BY id DESC LIMIT ?)
        """, (user_id, user_id, MAX_PER_USER)).fetchall()
        if not stale:
            return
        ids = [str(r[0]) for r in stale]
        c.execute(f"DELETE FROM memories WHERE id IN ({','.join('?' * len(ids))})", ids)
    try:
        _vec().delete(ids=ids)
    except Exception:
        pass


# ============================================================
# 检索
# ============================================================

def search(user_id, query, k=TOP_K):
    """按语义检索与当前问题最相关的记忆，返回 [(kind, content, score), ...]"""
    try:
        hits = _vec().similarity_search_with_relevance_scores(
            query, k=k, filter={"user_id": user_id})
    except Exception as e:
        print(f"  [记忆] 向量检索失败，回退到最近记忆：{e}")
        return [(kind, content, None) for kind, content in load(user_id, limit=k)]

    out = []
    for doc, score in hits:
        if score is None or score >= MIN_SCORE:      # 过滤不相关记忆
            out.append((doc.metadata.get("kind", ""), doc.page_content, score))
    return out


def load(user_id, limit=50):
    """按时间倒序读取记忆（用于 /memory 查看全部），返回 [(kind, content), ...]"""
    with _conn() as c:
        rows = c.execute(
            "SELECT kind, content FROM memories WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (user_id, limit)).fetchall()
    return list(reversed(rows))


# ============================================================
# 格式化与维护
# ============================================================

def format_memories(rows):
    """转成可注入系统提示词的文本。rows 为 (kind, content) 或 (kind, content, score)"""
    if not rows:
        return ""
    lines = ["[与当前问题相关的用户记忆]"]
    for r in rows:
        kind, content = r[0], r[1]
        lines.append(f"- {LABELS.get(kind, kind)}：{content}")
    return "\n".join(lines)


def count(user_id):
    with _conn() as c:
        return c.execute("SELECT COUNT(*) FROM memories WHERE user_id=?",
                         (user_id,)).fetchone()[0]


def clear(user_id):
    """清空该用户的全部记忆（两处存储同时清理）"""
    with _conn() as c:
        rows = c.execute("SELECT id FROM memories WHERE user_id=?", (user_id,)).fetchall()
        c.execute("DELETE FROM memories WHERE user_id=?", (user_id,))
    if rows:
        try:
            _vec().delete(ids=[str(r[0]) for r in rows])
        except Exception:
            pass
    return len(rows)
