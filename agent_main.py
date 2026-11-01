"""Agent 版入口：基于 LangGraph 的财报分析智能体"""
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
import agent

load_dotenv()


def main():
    print("=" * 56)
    print("财报分析智能体（Tesla / Apple 10-K）　输入 quit 退出")
    print("=" * 56)
    print("正在加载模型与向量库 ...")
    agent.warmup()
    print("就绪。")

    history = []          # 短期记忆：本次会话的对话历史
    while True:
        q = input("\n请输入问题：").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if not q:
            continue

        answer, trace = agent.run_agent(q, history=history)
        print(f"\n回答：{answer}")

        history += [HumanMessage(q), AIMessage(answer)]
        history = history[-10:]      # 保留最近 5 轮，控制上下文长度


if __name__ == "__main__":
    main()
