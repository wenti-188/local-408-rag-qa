"""
Flask后端：本地408 RAG问答Web服务
启动后访问 http://localhost 即可使用网页版问答
说明：当前使用Python内置SQLite零配置存储问答记录，生产环境切换MySQL只需替换DB_CONFIG即可
"""
import os
import sqlite3
import json
import random
from datetime import datetime
from flask import Flask, request, jsonify, render_template, Response
from langchain_ollama import ChatOllama

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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

# 初始化模型和我们自己手写的迷你向量库（启动时加载一次）
print("正在加载模型和知识库，请稍等...")
from minivectordb import MiniVectorDB
llm = ChatOllama(model="qwen2.5:7b", temperature=0.3, num_predict=250)
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
                # 按50-80字的小知识点切分，所有非空行都处理，不再只切短横线开头的列表
                lines = [line.strip() for line in content.split("\n") if line.strip() and not line.strip().startswith("#")]
                buffer = ""
                for line in lines:
                    if len(buffer) + len(line) < 80:
                        buffer += line + " "
                    else:
                        if buffer.strip():
                            all_texts.append(buffer.strip())
                        buffer = line + " "
                if buffer.strip():
                    all_texts.append(buffer.strip())
    vector_db.add_texts(all_texts)
    vector_db.save()

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

    def generate():
        full_answer = ""
        try:
            context = "\n---\n".join([doc for doc, score in vector_db.search(question, top_k=4)])
            messages = [
                {"role": "system", "content": "你是计算机专业复习辅导老师，覆盖408考研知识点、互联网面试八股文、C/C++/Java/Python编程语言、MySQL数据库、Web开发、机器学习等内容，严格根据参考资料回答：如果用户问的是具体知识点问题，先给核心结论，再讲原理和易错点，控制在250字内，不要瞎编参考资料里没有的内容。参考资料：" + context},
                {"role": "user", "content": question}
            ]
            for chunk in llm.stream(messages):
                token = chunk.content
                full_answer += token
                yield token
        except Exception as e:
            yield "⚠️ 本地大模型未启动，请先打开Ollama客户端再试。"
        elapsed = round(time.time() - start_time, 1)
        # 存数据库
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO chat_records (question, answer) VALUES (?, ?)", (question, full_answer))
        conn.commit()
        conn.close()

    return Response(generate(), mimetype="text/plain")

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
    参考下面的408知识点/真题，出一道单选题：稍微改动题目里的数字或者条件，不要完全照抄原题，考察的核心考点不变，题目要贴合考研408风格，同时写一句简短解析。
    严格输出JSON格式，不要输出任何其他解释文字，格式如下：
    {{
        "question": "题目内容",
        "options": {{"A": "选项A内容", "B": "选项B内容", "C": "选项C内容", "D": "选项D内容"}},
        "correct_answer": "正确选项的字母（A/B/C/D）",
        "explanation": "简短的解析说明，讲清楚正确答案的原理"
    }}
    参考内容：{random_doc[:400]}
    """
    raw_result = llm.invoke(quiz_prompt).content
    # 清洗模型输出的markdown包裹
    raw_result = raw_result.replace("```json", "").replace("```", "").strip()
    try:
        quiz = json.loads(raw_result)
        return jsonify(quiz)
    except:
        return jsonify({"error": "题目生成失败，请重试"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=False, threaded=True)
