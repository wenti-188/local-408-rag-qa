"""
大模型API网关服务（独立运行在8000端口，不影响原有80端口的复习系统）
实现：API Key鉴权、Token消耗统计、Agent工具调用
"""
import sqlite3
import time
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import ollama
from minivectordb import MiniVectorDB

DB_PATH = "408rag.db"
app = FastAPI(title="408RAG 本地大模型API网关")
api_key_header = APIKeyHeader(name="X-API-Key")
# 初始化向量库（复用原来手写的MiniVectorDB）
vector_db = MiniVectorDB(storage_file="408_vectordb.json")

# 初始化数据库表
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS api_keys (
        key TEXT PRIMARY KEY,
        name TEXT,
        created_at REAL
    )''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS usage_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        api_key TEXT,
        prompt_tokens INTEGER,
        completion_tokens INTEGER,
        total_tokens INTEGER,
        created_at REAL
    )''')
    cursor.execute("SELECT * FROM api_keys WHERE key = ?", ("test-key-12345",))
    if not cursor.fetchone():
        cursor.execute("INSERT INTO api_keys VALUES (?, ?, ?)", ("test-key-12345", "默认测试Key", time.time()))
    conn.commit()
    conn.close()

init_db()

def verify_api_key(key: str = Depends(api_key_header)):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM api_keys WHERE key = ?", (key,))
    if not cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=401, detail="无效的API Key")
    conn.close()
    return key

class ChatRequest(BaseModel):
    prompt: str

def rag_tool(query: str) -> str:
    """封装成Agent可调用的工具：检索408知识库"""
    results = vector_db.search(query, top_k=3)
    return "\n---\n".join([doc for doc, score in results])

@app.post("/v1/chat")
def chat(req: ChatRequest, api_key: str = Depends(verify_api_key)):
    """带Agent工具调用的聊天接口：模型自己判断要不要查知识库"""
    try:
        # 第一步：让模型先判断这个问题要不要查408资料
        check_res = ollama.chat(
            model="qwen2.5:7b",
            messages=[
                {"role": "system", "content": "判断下面用户的问题是否需要查询408考研知识点资料，只回答'是'或'否'，不要说别的。"},
                {"role": "user", "content": req.prompt}
            ]
        )
        need_rag = "是" in check_res["message"]["content"]
        context = ""
        if need_rag:
            context = rag_tool(req.prompt)
        # 第二步：根据判断结果，带上下文或者不带上下文，调用模型生成回答
        system_prompt = "你是408复习助手"
        if context:
            system_prompt += f"，参考以下408资料回答：{context}"
        res = ollama.chat(
            model="qwen2.5:7b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": req.prompt}
            ]
        )
        # 记录Token消耗
        prompt_tokens = res.get("prompt_eval_count", 0)
        completion_tokens = res.get("eval_count", 0)
        total_tokens = prompt_tokens + completion_tokens
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO usage_logs (api_key, prompt_tokens, completion_tokens, total_tokens, created_at) VALUES (?, ?, ?, ?, ?)",
                       (api_key, prompt_tokens, completion_tokens, total_tokens, time.time()))
        conn.commit()
        conn.close()
        return {
            "content": res["message"]["content"],
            "used_rag": need_rag,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"调用失败：{str(e)}")

@app.get("/")
def dev_console():
    html = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>开发者控制台</title>
    <style>
        body { font-family: "微软雅黑", sans-serif; background: #fafafa; color: #111; max-width: 700px; margin: 40px auto; padding: 0 20px; }
        h1 { text-align: center; font-size: 22px; margin-bottom: 8px; }
        .sub { text-align: center; color: #999; font-size: 13px; margin-bottom: 32px; }
        .card { background: white; border: 1px solid #eee; border-radius: 8px; padding: 24px; margin-bottom: 20px; }
        h2 { font-size: 16px; margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid #eee; }
        input { width: 100%; padding: 12px; border: 1px solid #ddd; border-radius: 6px; font-size: 14px; margin: 8px 0; box-sizing: border-box; }
        button { padding: 10px 20px; background: #111; color: white; border: none; border-radius: 6px; cursor: pointer; margin-top: 8px; }
        .result { margin-top: 16px; padding: 12px; background: #f5f5f5; border-radius: 6px; font-size: 14px; line-height: 1.8; display: none; }
        .dots { display: inline-block; margin-left: 8px; }
        .dots span { display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: #999; margin: 0 2px; animation: bounce 1.2s infinite; }
        .dots span:nth-child(2) { animation-delay: 0.2s; }
        .dots span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes bounce { 0%,80%,100% { transform: translateY(0); opacity:0.4; } 40% { transform: translateY(-6px); opacity:1; } }
        .back { text-align: center; margin-top: 24px; }
        .back a { color: #111; text-decoration: none; border-bottom: 1px solid #111; }
    </style>
</head>
<body>
    <h1>408本地大模型开发者控制台</h1>
    <div class="sub">完全本地运行 · Agent工具调用 · API鉴权 · Token用量统计</div>
    <div class="card">
        <h2>查询我的用量</h2>
        <input type="text" id="key" placeholder="测试Key：test-key-12345">
        <button onclick="query()">查询用量</button>
        <div class="result" id="res"></div>
    </div>
    <div class="card">
        <h2>接口说明</h2>
        <p style="font-size:14px; line-height:2; color:#333;">
        聊天接口：POST /v1/chat<br>
        请求头带 X-API-Key: 你的Key<br>
        请求体：{"prompt": "你的问题"}<br>
        返回内容：模型回答、是否调用了知识库、本次Token消耗
        </p>
    </div>
    <div class="back"><a href="http://localhost">← 返回复习助手</a></div>
    <script>
    async function query() {
        const key = document.getElementById('key').value.trim();
        const res = document.getElementById('res');
        res.style.display = 'block';
        res.innerHTML = '正在查询 <span class="dots"><span></span><span></span><span></span></span>';
        try {
            const r = await fetch('/v1/usage', {headers: {'X-API-Key': key}});
            if (!r.ok) throw new Error('无效Key');
            const data = await r.json();
            res.innerHTML = '✅ 查询成功<br>累计调用：' + data.call_count + '次<br>累计消耗Token：' + data.total_tokens_used + '个';
        } catch(e) {
            res.innerHTML = '❌ 查询失败：' + e.message;
        }
    }
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html)

@app.get("/v1/usage")
def get_usage(api_key: str = Depends(verify_api_key)):
    """查询当前Key的总Token消耗和调用次数"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*), SUM(total_tokens) FROM usage_logs WHERE api_key = ?", (api_key,))
    count, total = cursor.fetchone()
    conn.close()
    return {
        "call_count": count,
        "total_tokens_used": total or 0
    }

if __name__ == "__main__":
    import uvicorn
    print("✅ API网关已启动，访问 http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
