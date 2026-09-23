"""程序入口:启动财报问答助手"""
from dotenv import load_dotenv
import rag_pipeline

load_dotenv()


def main():
    # 准备向量库和大模型
    vectorstore = rag_pipeline.get_vectorstore()
    llm = rag_pipeline.get_llm()

    print("\n" + "=" * 60)
    print("财报问答助手已就绪!输入你的问题，输入 quit 退出。")
    print("=" * 60)

    while True:
        question = input("\n请输入问题：").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("再见!")
            break
        if not question:
            continue
        print("正在思考...")
        answer = rag_pipeline.ask(vectorstore, llm, question)
        print(f"\n回答:{answer}")


if __name__ == "__main__":
    main()