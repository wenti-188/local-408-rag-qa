"""
Flask后端：本地408 RAG问答Web服务
启动后访问 http://localhost:5000 即可使用网页版问答
说明：当前使用Python内置SQLite零配置存储问答记录，生产环境切换MySQL只需替换DB_CONFIG即可
"""
import os
import sqlite3
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
DB_PATH = os.path.join(BASE_DIR, "chat_history.db")

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

# 初始化数据库（首次运行自动建表）
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# 初始化模型和向量库（启动时加载一次）
print("正在加载模型和知识库，请稍等...")
embeddings = OllamaEmbeddings(model="bge-m3")
llm = ChatOllama(model="qwen2.5:7b", temperature=0.3, num_predict=400)
vector_store = Chroma(
    collection_name="408_knowledge",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR
)
prompt = ChatPromptTemplate.from_template("""
你是考研408专业课的辅导老师，严格根据参考资料回答问题，不要瞎编资料里没有的内容。如果参考资料里没有，就说"知识库中暂时没有这部分内容"。

回答紧凑有层次：先给核心结论，再简单讲清楚原理和关键性质，最后补1-2句常考易错点，不要写多余的废话，不要过度展开，控制在300字左右。

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
print("✅ 服务加载完成，访问 http://localhost:5000 即可使用")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"answer": "请输入问题", "elapsed_time": 0}), 400
    import time
    start_time = time.time()
    answer = rag_chain.invoke(question)
    elapsed_time = round(time.time() - start_time, 1)
    # 自动把问答记录存进数据库
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chat_records (question, answer) VALUES (?, ?)",
        (question, answer)
    )
    conn.commit()
    conn.close()
    return jsonify({"answer": answer, "elapsed_time": elapsed_time})

@app.route("/api/history")
def history():
    # 查询最近20条历史问答记录
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT question, answer, create_time FROM chat_records ORDER BY id DESC LIMIT 20")
    records = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify({"records": records})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=False, threaded=True)
