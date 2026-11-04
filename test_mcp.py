"""单独测试 MCP Server：不经过 Agent，直接按协议连上去

演示 MCP 的三个基本动作：initialize（握手）→ list_tools（列工具）→ call_tool（调用）
"""
import os
import sys
import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    # ---------- ① 描述如何启动 Server ----------
    # sys.executable = 当前虚拟环境的 python.exe，保证子进程用同一个环境
    params = StdioServerParameters(
        command=sys.executable,
        args=[os.path.abspath("mcp_server.py")],
    )

    # ---------- ② 启动子进程，建立 stdio 通道 ----------
    async with stdio_client(params) as (read, write):
        # ---------- ③ 在通道上建立 MCP 会话 ----------
        async with ClientSession(read, write) as session:
            await session.initialize()                    # 握手
            print("✓ 握手成功\n")

            tools = await session.list_tools()            # 询问有哪些工具
            print("Server 提供的工具：")
            for t in tools.tools:
                print(f"  - {t.name}：{t.description.splitlines()[0]}")

            # ---------- ④ 调用工具 ----------
            print("\n[测试1] query_financials")
            r = await session.call_tool(
                "query_financials", {"item": "total revenues", "company": "Tesla"})
            print(r.content[0].text)

            print("\n[测试2] calculate（批量）")
            r = await session.call_tool(
                "calculate", {"expressions": "(94827-97690)/97690*100; (12771/6035)**0.5-1"})
            print(r.content[0].text)

            print("\n[测试3] get_breakdown")
            r = await session.call_tool(
                "get_breakdown", {"parent_item": "Total automotive revenues"})
            print(r.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())
