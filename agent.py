"""基于 LangGraph 的财报分析智能体（含长短期记忆）

状态图结构：
    START → load_memory → analyst ──(有工具调用)──> tools ──> analyst
                             └──────(无工具调用)──> save_memory → END

短期记忆：LangGraph checkpointer，按 thread_id 保存本次会话状态
长期记忆：向量检索 + SQLite（见 memory.py），按 user_id 跨会话保存
"""
import re
import json
import threading
from typing import Annotated, TypedDict

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

try:                                    # 新旧版本 langgraph 类名不同
    from langgraph.checkpoint.memory import InMemorySaver as _Saver
except ImportError:
    from langgraph.checkpoint.memory import MemorySaver as _Saver

import rag_pipeline
import table_db
import memory
import config

# ============================================================
# 资源加载（加锁，防止 ToolNode 并发初始化冲突）
# ============================================================

_VS = None
_VS_LOCK = threading.Lock()


def _vs():
    global _VS
    if _VS is None:
        with _VS_LOCK:
            if _VS is None:
                _VS = rag_pipeline.get_vectorstore()
    return _VS


def warmup():
    """启动时预热向量库与重排模型"""
    _vs()
    rag_pipeline.get_reranker()


# ============================================================
# 工具定义
# ============================================================

@tool
def search_filings(query: str, company: str = "") -> str:
    """在财报原文中检索相关段落。适合风险、业务描述、管理层解释等文字类问题。
    company 可选 "Tesla" 或 "Apple"，留空则检索全部公司。"""
    flt = {"company": company} if company else None
    docs = _vs().similarity_search(query, k=config.RETRIEVE_TOP_K, filter=flt)
    if not docs:
        return "未检索到相关内容"
    scores = rag_pipeline.get_reranker().predict(
        [[query, d.page_content] for d in docs])
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    out = []
    for _, d in ranked[:config.RERANK_TOP_K]:
        m = d.metadata
        out.append(f"[{m.get('company')} {m.get('section')} 第{m.get('page')}页]\n"
                   f"{d.page_content[:600]}")
    return "\n\n".join(out)


@tool
def query_financials(item: str, company: str = "") -> str:
    """从财报表格的结构化数据库中精确查询某财务科目的多年数值。
    数字类问题应优先使用本工具，而非从原文中猜测。
    item 使用英文科目名，如 "total revenues"、"net income"、"gross profit"。
    company 可选 "Tesla" 或 "Apple"，留空则查询全部公司。"""
    rows = table_db.query_trend(item, company=company or None, limit=15)
    return table_db.format_results(rows) or f"未找到科目：{item}"


@tool
def get_breakdown(parent_item: str) -> str:
    """查询某汇总科目由哪些子项构成及各子项数值。
    适用于"哪个子项拖累最多""构成占比如何"这类分析。
    parent_item 使用完整汇总科目名，如 "Total automotive revenues"。"""
    rows = table_db.query_children(parent_item)
    if not rows:
        return f"未找到 {parent_item} 的构成明细"
    seen, lines = set(), []
    for item, year, value, page in rows:
        if (item, year) not in seen:
            seen.add((item, year))
            lines.append(f"{item} | {year}年 | {value:,.0f} | 第{page}页")
    return "\n".join(lines)


@tool
def calculate(expression: str) -> str:
    """计算数学表达式，用于增长率、占比、差额、复合增长率等。
    示例："(94827-97690)/97690*100"、"(12771/6035)**0.5-1"。
    涉及计算时必须调用本工具，不要心算。"""
    if not re.fullmatch(r"[\d\s\.\+\-\*/\(\)eE]+", expression):
        return "表达式包含非法字符，仅允许数字与 + - * / ( ) **"
    try:
        value = eval(expression, {"__builtins__": {}})
        return f"{expression} = {value:.4f}"
    except Exception as e:
        return f"计算出错：{e}"


TOOLS = [search_filings, query_financials, get_breakdown, calculate]

ANALYST_PROMPT = """你是一个财报分析智能体，可查询特斯拉（Tesla）与苹果（Apple）的 10-K 年报。

首先判断用户这句话属于哪种情况：
A. 陈述偏好或背景（如"我关注能源业务""回答简洁点"）——不要调用任何工具，
   用一句话确认收到即可，例如"好的，已记住"。
B. 查询财报信息——按下面的原则调用工具作答。

查询类问题的工作原则：
1. 数字类问题必须调用 query_financials 获取精确数值，不得凭记忆作答
2. 任何计算（增长率、占比、CAGR）必须调用 calculate，不要心算
3. 分析构成、占比、贡献度时使用 get_breakdown
4. 风险、业务描述等文字类问题使用 search_filings
5. 回答末尾标注数据来源（公司、章节、页码）
6. 工具查不到的内容如实说明，不要编造
7. 只回答用户当前问的问题，不要主动重复或扩展上一轮的内容"""

EXTRACT_PROMPT = """请从下面这轮对话中，提取值得长期记住、对今后对话有帮助的信息。

只提取以下三类：
- profile：用户关注的公司、业务或指标（如"关注特斯拉的毛利率"）
- preference：用户对回答方式的偏好（如"希望回答简洁""希望用表格"）
- finding：本轮分析得出的关键结论，须包含具体数字与年份

要求：
- 只提取明确的信息，不要推测；闲聊或无价值的内容不提取
- 每条一句话，最多 3 条
- 没有值得记住的内容就返回 []
- 只输出 JSON 数组，不要任何其他文字

格式示例：[{{"kind": "profile", "content": "关注特斯拉能源业务"}}]

用户问题：{q}
助手回答：{a}"""


# ============================================================
# 状态定义
# ============================================================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # 短期记忆（由 checkpointer 持久化）
    memory: str                               # 本轮检索到的长期记忆文本
    user_id: str


# ============================================================
# 辅助函数
# ============================================================

def _recent_turns(msgs, k=5):
    """只保留最近 k 轮。按用户提问切分，避免把工具调用与其结果切开"""
    idx = [i for i, m in enumerate(msgs) if isinstance(m, HumanMessage)]
    return msgs[idx[-k]:] if len(idx) > k else msgs


def _parse_json_list(text):
    """从模型输出中稳健解析 JSON 数组"""
    text = text.strip().replace("```json", "").replace("```", "")
    s, e = text.find("["), text.rfind("]")
    if s == -1 or e == -1:
        return []
    try:
        data = json.loads(text[s:e + 1])
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


# ============================================================
# 节点
# ============================================================

def load_memory_node(state: AgentState):
    """按当前问题语义检索相关的长期记忆（而非全量注入）"""
    msgs = state["messages"]
    question = next((m.content for m in reversed(msgs)
                     if isinstance(m, HumanMessage)), "")
    user_id = state.get("user_id", "default")

    rows = memory.search(user_id, question, k=memory.TOP_K)
    if rows:
        shown = ", ".join(f"{r[2]:.2f}" for r in rows if r[2] is not None)
        print(f"  [记忆] 从 {memory.count(user_id)} 条中检索到 {len(rows)} 条相关"
              + (f"（相关度 {shown}）" if shown else ""))
    return {"memory": memory.format_memories(rows)}


def analyst_node(state: AgentState):
    """分析员：结合长期记忆与近期对话，决定调用工具还是作答"""
    system = ANALYST_PROMPT
    if state.get("memory"):
        system += ("\n\n" + state["memory"] +
                   "\n\n请结合以上记忆理解用户的关注点与偏好；"
                   "但历史结论只作参考，涉及数字时仍须通过工具重新核实。")
    llm = rag_pipeline.get_llm().bind_tools(TOOLS)
    msgs = [SystemMessage(system)] + _recent_turns(state["messages"])
    return {"messages": [llm.invoke(msgs)]}


def should_continue(state: AgentState) -> str:
    """条件边：请求调用工具则去 tools，否则去保存记忆"""
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "save_memory"


def save_memory_node(state: AgentState):
    """从本轮对话中提取值得长期保存的信息"""
    msgs = state["messages"]
    question = next((m.content for m in reversed(msgs)
                     if isinstance(m, HumanMessage)), "")
    answer = msgs[-1].content if msgs else ""
    try:
        resp = rag_pipeline.get_llm().invoke(
            [HumanMessage(EXTRACT_PROMPT.format(q=question, a=answer[:1500]))])
        items = _parse_json_list(resp.content)
        n = memory.add_many(state.get("user_id", "default"), items)
        if n:
            print(f"  [记忆] 新增 {n} 条长期记忆")
    except Exception as e:
        print(f"  [记忆] 提取失败，已跳过：{e}")
    return {}


# ============================================================
# 构图
# ============================================================

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("load_memory", load_memory_node)
    g.add_node("analyst", analyst_node)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_node("save_memory", save_memory_node)

    g.add_edge(START, "load_memory")
    g.add_edge("load_memory", "analyst")
    g.add_conditional_edges("analyst", should_continue,
                            {"tools": "tools", "save_memory": "save_memory"})
    g.add_edge("tools", "analyst")          # 工具结果回到分析员，形成循环
    g.add_edge("save_memory", END)

    return g.compile(checkpointer=_Saver())   # checkpointer 提供短期记忆


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def run_agent(question, user_id="default", thread_id="default", verbose=True):
    """
    执行一次问答，返回 (最终回答, 本轮工具调用轨迹)
    thread_id：会话标识，同一 thread 内共享短期记忆
    user_id  ：用户标识，决定读写哪个用户的长期记忆
    """
    cfg = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}
    result = get_graph().invoke(
        {"messages": [HumanMessage(question)], "user_id": user_id}, cfg)

    msgs = result["messages"]
    last_human = max(i for i, m in enumerate(msgs) if isinstance(m, HumanMessage))
    trace = []
    for m in msgs[last_human:]:
        for call in getattr(m, "tool_calls", None) or []:
            trace.append({"tool": call["name"], "args": call["args"]})
            if verbose:
                print(f"  → 调用 {call['name']}({call['args']})")
    return msgs[-1].content, trace
