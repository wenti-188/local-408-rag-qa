"""
最小可运行的本地RAG链路测试脚本
验证：bge-m3向量嵌入 -> Chroma本地向量库存储 -> 相似性检索 -> qwen2.5:7b生成回答
"""
import os
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 项目路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

# 1. 初始化：嵌入模型用本地bge-m3，大模型用本地qwen2.5:7b
print("正在加载本地模型...")
embeddings = OllamaEmbeddings(model="bge-m3")
llm = ChatOllama(model="qwen2.5:7b", temperature=0.3)

# 2. 测试用的408知识点（后续替换成你真实的408笔记文件）
test_docs = [
    "进程是资源分配的最小单位，线程是CPU调度的最小单位；进程有独立地址空间，线程共享进程的地址空间。",
    "快速排序的平均时间复杂度是O(nlogn)，最坏情况是O(n²)，属于不稳定排序；堆排序平均和最坏都是O(nlogn)，空间复杂度O(1)。",
    "TCP是面向连接的可靠传输协议，通过三次握手建立连接，四次挥手释放连接；UDP是无连接的不可靠协议，适合实时视频、语音场景。",
    "数据库的ACID特性：原子性(Atomicity)、一致性(Consistency)、隔离性(Isolation)、持久性(Durability)。",
    "虚拟页式存储管理中，缺页中断会将所需页面从外存调入内存；常见的页面置换算法有FIFO、LRU、OPT。"
]
test_ids = [f"test_chunk_{i}" for i in range(len(test_docs))]

# 3. 初始化/加载Chroma本地向量库
print("正在初始化本地向量库...")
vector_store = Chroma(
    collection_name="408_knowledge",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR
)

# 如果向量库是空的，先把测试知识点写进去
if len(vector_store.get()["ids"]) == 0:
    print("向量库为空，正在写入测试知识点...")
    vector_store.add_texts(texts=test_docs, ids=test_ids)

# 4. 定义RAG提示词模板
prompt = ChatPromptTemplate.from_template("""
你是一个考研408辅导助手，请严格根据下面提供的参考资料回答用户问题。
如果参考资料里没有相关信息，就如实说你不知道，不要编造答案。

参考资料：
{context}

用户问题：{question}
回答：
""")

# 5. 构建RAG链：检索 -> 拼接提示词 -> 调用大模型 -> 解析输出
rag_chain = (
    {"context": lambda q: "\n---\n".join([doc.page_content for doc in vector_store.similarity_search(q, k=2)]),
     "question": lambda q: q}
    | prompt
    | llm
    | StrOutputParser()
)

# 6. 跑一个测试问题
test_question = "快速排序的时间复杂度是多少？它稳定吗？"
print(f"\n测试问题：{test_question}")
print("正在检索并生成回答（首次加载大模型需要等待10-30秒）...\n")
answer = rag_chain.invoke(test_question)
print("="*50)
print("回答结果：")
print(answer)
print("="*50)
print("\n✅ 全链路跑通！说明：向量库正常、本地大模型正常、RAG流程正常。")
