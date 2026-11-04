"""Agent 入口：LangGraph 财报分析智能体（MCP 工具 + 规划 + 长短期记忆）

命令：
    /memory   查看当前用户的全部长期记忆
    /forget   清空当前用户的长期记忆
    /tools    查看当前可用工具及其来源
    quit      退出

本文件为异步入口：MCP 通过 stdio 与子进程通信，需在事件循环中运行。
"""
import uuid
import asyncio
from dotenv import load_dotenv

import agent
import memory

load_dotenv()

USER_ID = "han"     # 用户标识：决定读写谁的长期记忆


async def main():
    print("=" * 60)
    print("财报分析智能体（Tesla / Apple 10-K）")
    print("命令：/memory 查看记忆　/forget 清空记忆　/tools 查看工具　quit 退出")
    print("=" * 60)
    print("正在加载模型、向量库与 MCP 服务 ...")

    await agent.warmup()           # 预热：向量库 + 重排模型 + 连接 MCP Server

    thread_id = str(uuid.uuid4())  # 每次启动为新会话（短期记忆清零）
    print(f"就绪。用户：{USER_ID}　会话：{thread_id[:8]}　"
          f"长期记忆 {memory.count(USER_ID)} 条\n")

    while True:
        try:
            q = input("\n请输入问题：").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if q.lower() in ("quit", "exit", "q"):
            break
        if not q:
            continue

        if q == "/memory":
            rows = memory.load(USER_ID, limit=100)
            print(memory.format_memories(rows) or "暂无长期记忆")
            print(f"（共 {memory.count(USER_ID)} 条）")
            continue

        if q == "/forget":
            n = memory.clear(USER_ID)
            print(f"已清空 {n} 条长期记忆")
            continue

        if q == "/tools":
            local = {t.name for t in agent.LOCAL_TOOLS}
            for t in agent.get_tools():
                src = "本地" if t.name in local else "MCP"
                print(f"  [{src}] {t.name}")
            continue

        try:
            answer, trace = await agent.arun_agent(
                q, user_id=USER_ID, thread_id=thread_id)
            print(f"\n回答：{answer}")
        except Exception as e:
            print(f"\n处理出错：{e}")

    print("再见。")


if __name__ == "__main__":
    asyncio.run(main())
