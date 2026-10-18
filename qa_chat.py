"""
本地RAG问答脚本：纯离线运行，基于知识库内容回答问题
输入问题后自动检索相关知识点，调用本地qwen2.5:7b生成回答
输入 exit 或 退出 即可结束程序
"""
import os
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

print("正在加载本地模型和知识库，请稍等...")
# 加载本地模型
embeddings = OllamaEmbeddings(model="bge-m3")
llm = ChatOllama(model="qwen2.5:7b", temperature=0.3)
vector_store = Chroma(
    collection_name="408_knowledge",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR
)

# RAG提示词
prompt = ChatPromptTemplate.from_template("""
你是考研408专业课辅导助手，必须严格根据下面提供的参考资料回答用户问题。
如果参考资料里没有相关内容，就直接说"知识库中暂时没有这部分内容"，绝对不要编造答案。
回答要简洁准确，适合考研复习场景。

参考资料：
{context}

用户问题：{question}
回答：
""")

rag_chain = (
    {"context": lambda q: "\n---\n".join([doc.page_content for doc in vector_store.similarity_search(q, k=3)]),
     "question": lambda q: q}
    | prompt
    | llm
    | StrOutputParser()
)

print("="*60)
print("✅ 408本地离线问答系统已就绪！")
print("输入你的408相关问题开始提问，输入 exit 或 退出 结束程序。")
print("="*60)

while True:
    question = input("\n你的问题：").strip()
    if question.lower() in ["exit", "quit", "退出"]:
        print("再见！")
        break
    if not question:
        continue
    print("\n正在检索并生成回答...\n")
    answer = rag_chain.invoke(question)
    print("回答：")
    print(answer)
