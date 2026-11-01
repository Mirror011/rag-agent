"""基于 LangGraph 的财报分析智能体

状态图结构：
    analyst ──(有工具调用)──> tools ──> analyst
       └──────(无工具调用)──> END
"""
import re
from typing import Annotated, TypedDict

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

import rag_pipeline
import table_db
import config

import threading

_VS = None
_VS_LOCK = threading.Lock()


def _vs():
    """向量库懒加载，加锁防止 ToolNode 并发初始化时冲突"""
    global _VS
    if _VS is None:
        with _VS_LOCK:
            if _VS is None:                       # 双重检查
                _VS = rag_pipeline.get_vectorstore()
    return _VS


def warmup():
    """启动时预热向量库与重排模型，避免首次并发调用时竞争初始化"""
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

工作原则：
1. 数字类问题必须调用 query_financials 获取精确数值，不得凭记忆作答
2. 任何计算（增长率、占比、CAGR）必须调用 calculate，不要心算
3. 分析构成、占比、贡献度时使用 get_breakdown
4. 风险、业务描述等文字类问题使用 search_filings
5. 回答末尾标注数据来源（公司、章节、页码）
6. 工具查不到的内容如实说明，不要编造"""


# ============================================================
# 状态定义
# ============================================================

class AgentState(TypedDict):
    """图在各节点之间传递的状态"""
    messages: Annotated[list, add_messages]


# ============================================================
# 节点
# ============================================================

def analyst_node(state: AgentState):
    """分析员：根据当前消息决定调用工具还是直接作答"""
    llm = rag_pipeline.get_llm().bind_tools(TOOLS)
    msgs = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in msgs):
        msgs = [SystemMessage(ANALYST_PROMPT)] + msgs
    return {"messages": [llm.invoke(msgs)]}


def should_continue(state: AgentState) -> str:
    """条件边：模型若请求调用工具则转向 tools，否则结束"""
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


# ============================================================
# 构图
# ============================================================

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("analyst", analyst_node)
    g.add_node("tools", ToolNode(TOOLS))

    g.add_edge(START, "analyst")
    g.add_conditional_edges("analyst", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "analyst")          # 工具结果回到分析员，形成循环
    return g.compile()


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def run_agent(question, history=None, verbose=True):
    """执行一次问答，返回 (最终回答, 工具调用轨迹)"""
    msgs = (history or []) + [HumanMessage(question)]
    result = get_graph().invoke({"messages": msgs}, {"recursion_limit": 15})

    trace = []
    for m in result["messages"]:
        for call in getattr(m, "tool_calls", None) or []:
            trace.append({"tool": call["name"], "args": call["args"]})
            if verbose:
                print(f"  → 调用 {call['name']}({call['args']})")
    return result["messages"][-1].content, trace
