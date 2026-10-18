"""
知识库自动导入脚本：读取knowledge文件夹下所有md/txt文件，切分后向量化存入Chroma向量库
支持增量导入，不会重复导入已经存在的内容
"""
import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

print("=== 开始导入知识库 ===")
# 1. 加载本地bge-m3嵌入模型
embeddings = OllamaEmbeddings(model="bge-m3")
vector_store = Chroma(
    collection_name="408_knowledge",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR
)

# 2. 读取knowledge目录下所有md/txt文件
all_docs = []
for filename in os.listdir(KNOWLEDGE_DIR):
    if filename.endswith(".md") or filename.endswith(".txt"):
        file_path = os.path.join(KNOWLEDGE_DIR, filename)
        print(f"正在读取文件：{filename}")
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        doc = Document(page_content=content, metadata={"source_file": filename})
        all_docs.append(doc)

print(f"\n共读取到 {len(all_docs)} 个原始文档")

# 3. 切分文本成适合检索的小块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n## ", "\n### ", "\n", "。", "！", "？", ""]
)
split_docs = text_splitter.split_documents(all_docs)
print(f"切分后共 {len(split_docs)} 个文本块")

# 4. 导入向量库
vector_store.add_documents(split_docs)
print(f"\n✅ 导入完成！当前知识库总共有 {len(vector_store.get()['ids'])} 个知识片段")
print("你现在可以运行 qa_chat.py 开始离线问答了。")
