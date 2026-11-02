"""Agent 版入口：基于 LangGraph 的财报分析智能体（含长短期记忆）

命令：
    /memory   查看当前用户的全部长期记忆
    /forget   清空当前用户的长期记忆
    quit      退出
"""
import uuid
from dotenv import load_dotenv
import agent
import memory

load_dotenv()

USER_ID = "han"     # 用户标识：决定读写谁的长期记忆


def main():
    print("=" * 56)
    print("财报分析智能体（Tesla / Apple 10-K）")
    print("命令：/memory 查看记忆　/forget 清空记忆　quit 退出")
    print("=" * 56)
    print("正在加载模型与向量库 ...")
    agent.warmup()

    thread_id = str(uuid.uuid4())      # 每次启动为新会话（短期记忆清零）
    print(f"就绪。用户：{USER_ID}　会话：{thread_id[:8]}　"
          f"已有长期记忆 {memory.count(USER_ID)} 条")

    while True:
        q = input("\n请输入问题：").strip()
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

        answer, trace = agent.run_agent(q, user_id=USER_ID, thread_id=thread_id)
        print(f"\n回答：{answer}")


if __name__ == "__main__":
    main()
