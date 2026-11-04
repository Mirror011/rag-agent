"""MCP Server：把财报结构化数据能力封装成标准 MCP 服务

提供 3 个工具：query_financials / get_breakdown / calculate
运行方式：由 Agent 以子进程方式自动启动，通过 stdio 通信，平时不需要手动运行。

注意：stdio 模式下，标准输出（stdout）是协议通道，专门用来传 MCP 消息。
     任何 print 到 stdout 的内容都会破坏通信，日志只能写到 stderr。
"""
import os
import re
import sys
import contextlib

# ---------- ① 固定工作目录 ----------
# Agent 启动本进程时，工作目录不一定是项目根目录；
# 而 config.py 和 financial_data.db 都按相对路径查找，所以先切换过来。
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)

from mcp.server.fastmcp import FastMCP

# ---------- ② 导入项目模块时屏蔽 stdout ----------
# table_db 会导入 pymupdf，后者可能打印提示信息；
# 把这期间的输出重定向到 stderr，避免污染协议通道。
with contextlib.redirect_stdout(sys.stderr):
    import table_db

# ---------- ③ 创建 Server ----------
# "financial-data" 是服务名，客户端连接后能看到
mcp = FastMCP("financial-data")


# ---------- ④ 注册工具 ----------
# @mcp.tool() 会自动读取：函数名 → 工具名；类型标注 → 参数格式；
# 文档字符串 → 工具说明（模型靠这段话判断何时调用）

@mcp.tool()
def query_financials(item: str, company: str = "") -> str:
    """从财报表格的结构化数据库中精确查询某财务科目的多年数值。
    数字类问题应优先使用本工具，而非从原文中猜测。
    item 使用英文科目名，如 "total revenues"、"net income"、"gross profit"。
    company 可选 "Tesla" 或 "Apple"，留空则查询全部公司。"""
    rows = table_db.query_trend(item, company=company or None, limit=15)
    return table_db.format_results(rows) or f"未找到科目：{item}"


@mcp.tool()
def get_breakdown(parent_item: str) -> str:
    """查询某汇总科目由哪些子项构成及各子项数值。
    适用于"哪个子项拖累最多""构成占比如何"这类分析。
    parent_item 使用完整英文汇总科目名，如 "Total automotive revenues"。"""
    rows = table_db.query_children(parent_item)
    if not rows:
        return f"未找到 {parent_item} 的构成明细"
    seen, lines = set(), []
    for item, year, value, page in rows:
        if (item, year) not in seen:
            seen.add((item, year))
            lines.append(f"{item} | {year}年 | {value:,.0f} | 第{page}页")
    return "\n".join(lines)


@mcp.tool()
def calculate(expressions: str) -> str:
    """计算一个或多个数学表达式，用于增长率、占比、差额、复合增长率等。
    多个表达式用分号 ; 分隔，一次性算完，不要分多次调用。
    示例："(94827-97690)/97690*100; (12771/6035)**0.5-1"
    涉及计算时必须调用本工具，不要心算。"""
    results = []
    for expr in expressions.split(";"):
        expr = expr.strip()
        if not expr:
            continue
        # 白名单校验：只允许数字和运算符，防止执行任意代码
        if not re.fullmatch(r"[\d\s\.\+\-\*/\(\)eE]+", expr):
            results.append(f"{expr} → 含非法字符，已跳过")
            continue
        try:
            results.append(f"{expr} = {eval(expr, {'__builtins__': {}}):.4f}")
        except Exception as e:
            results.append(f"{expr} → 计算出错：{e}")
    return "\n".join(results) if results else "未提供有效表达式"


# ---------- ⑤ 启动 ----------
if __name__ == "__main__":
    mcp.run(transport="stdio")
