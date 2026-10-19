"""
Flask后端：本地408 RAG问答Web服务
启动后访问 http://localhost:5000 即可使用网页版问答
说明：当前使用Python内置SQLite零配置存储问答记录，生产环境切换MySQL只需替换DB_CONFIG即可
"""
import os
import sqlite3
import json
import random
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
    # 新增错题表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS wrong_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            options TEXT NOT NULL,
            correct_answer TEXT NOT NULL,
            user_answer TEXT NOT NULL,
            knowledge_point TEXT,
            create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# 初始化模型和我们自己手写的迷你向量库（启动时加载一次）
print("正在加载模型和知识库，请稍等...")
from minivectordb import MiniVectorDB
import os
llm = ChatOllama(model="qwen2.5:7b", temperature=0.3, num_predict=400)
vector_db = MiniVectorDB(storage_file="408_vectordb.json")

# 如果库是空的，自动导入知识库文件
if len(vector_db.documents) == 0:
    print("首次运行，正在导入408知识点到自定义向量库...")
    knowledge_dir = os.path.join(BASE_DIR, "knowledge")
    all_texts = []
    for filename in os.listdir(knowledge_dir):
        if filename.endswith(".md"):
            with open(os.path.join(knowledge_dir, filename), "r", encoding="utf-8") as f:
                content = f.read()
                paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 20]
                all_texts.extend(paragraphs)
    vector_db.add_texts(all_texts)
    vector_db.save()

prompt = ChatPromptTemplate.from_template("""
你是考研408专业课的辅导老师，严格根据参考资料回答问题，不要瞎编资料里没有的内容。如果参考资料里没有，就说"知识库中暂时没有这部分内容"。

回答紧凑有层次：先给核心结论，再简单讲清楚原理和关键性质，最后补1-2句常考易错点，不要写多余的废话，不要过度展开，控制在300字左右。

参考资料：
{context}

用户问题：{question}
回答：
""")
rag_chain = (
    {"context": lambda q: "\n---\n".join([doc for doc, score in vector_db.search(q, top_k=3)]),
     "question": lambda q: q}
    | prompt
    | llm
    | StrOutputParser()
)
print("✅ 服务加载完成，访问 http://localhost 即可使用")

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

# ===== 刷题相关接口 =====
@app.route("/api/generate-quiz")
def generate_quiz():
    # 随机抽一个知识点，生成一道408单选题
    random_doc = random.choice(vector_db.documents)
    quiz_prompt = f"""
    根据下面这个408知识点，出一道单选题，题目要考察核心考点，不要出偏题。
    严格输出JSON格式，不要输出任何其他解释文字，格式如下：
    {{
        "question": "题目内容",
        "options": {{"A": "选项A内容", "B": "选项B内容", "C": "选项C内容", "D": "选项D内容"}},
        "correct_answer": "正确选项的字母（A/B/C/D）",
        "knowledge_point": "这个题对应的知识点名称"
    }}
    知识点内容：{random_doc[:300]}
    """
    raw_result = llm.invoke(quiz_prompt).content
    # 清洗模型输出的markdown包裹
    raw_result = raw_result.replace("```json", "").replace("```", "").strip()
    try:
        quiz = json.loads(raw_result)
        return jsonify(quiz)
    except:
        return jsonify({"error": "题目生成失败，请重试"}), 500

@app.route("/api/submit-quiz", methods=["POST"])
def submit_quiz():
    data = request.get_json()
    question = data.get("question")
    options = data.get("options")
    correct_answer = data.get("correct_answer")
    user_answer = data.get("user_answer")
    knowledge_point = data.get("knowledge_point", "")
    is_correct = (user_answer == correct_answer)
    # 如果答错了，存到错题表
    if not is_correct:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO wrong_questions (question, options, correct_answer, user_answer, knowledge_point) VALUES (?, ?, ?, ?, ?)",
            (question, json.dumps(options, ensure_ascii=False), correct_answer, user_answer, knowledge_point)
        )
        conn.commit()
        conn.close()
    return jsonify({"is_correct": is_correct, "correct_answer": correct_answer})

@app.route("/api/wrong-list")
def wrong_list():
    # 查询所有错题
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM wrong_questions ORDER BY id DESC")
    records = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify({"records": records})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=False, threaded=True)
