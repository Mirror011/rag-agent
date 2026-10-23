import rag_pipeline

vs = rag_pipeline.get_vectorstore()
docs = vs.similarity_search("total revenue", k=3)
for i, d in enumerate(docs, 1):
    print(f"\n--- 结果 {i} ---")
    print("元数据:", d.metadata)
    print("内容:", d.page_content[:150])