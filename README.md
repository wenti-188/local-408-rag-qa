# 基于本地大模型的408知识点RAG检索问答系统

## 项目简介
这是一个完全离线运行、不依赖任何云端API的本地RAG（检索增强生成）问答系统，针对考研408计算机学科专业基础（数据结构、计算机组成原理、操作系统、计算机网络）构建专属知识库。系统全程在本地完成文档加载、文本切分、向量存储、相似性检索、大模型生成回答全流程，数据不出本机，无联网成本。

## 技术栈
- 大模型：Ollama + Qwen2.5:7B（本地部署）
- 向量模型：BGE-M3（本地文本嵌入）
- RAG框架：LangChain
- 向量数据库：Chroma（本地持久化存储）
- 后端框架：Flask
- 数据存储：SQLite/MySQL（存储问答历史记录）
- 前端：原生HTML/CSS/JS聊天交互界面
- 开发环境：Python 3.13

## 核心功能
1.  本地知识库管理：支持导入Markdown、PDF格式的复习资料，自动切分文本块、向量化存入本地Chroma向量库，支持增量导入。
2.  离线问答：用户输入408相关问题时，系统自动从知识库检索最相关的知识点片段，拼接提示词调用本地Qwen2.5:7B模型生成准确回答，全程不调用云端大模型API。
3.  问答记录持久化：所有用户提问和系统回答自动存入数据库，支持历史记录查询。
4.  轻量Web交互界面：浏览器打开即可使用，类ChatGPT聊天交互体验。

## 项目结构
```
├── app.py                 # Flask后端服务主入口
├── ingest_knowledge.py    # 知识库导入脚本：读取文档、切分、向量化入库
├── qa_chat.py             # 命令行版问答程序
├── knowledge/             # 知识库文档目录，放入md/pdf格式的408笔记
├── chroma_db/             # Chroma向量库持久化目录
├── templates/
│   └── index.html         # 前端聊天界面
└── venv/                  # Python虚拟环境
```

## 本地启动步骤
1.  安装Ollama客户端，拉取所需模型：
    ```bash
    ollama pull qwen2.5:7b
    ollama pull bge-m3
    ```
2.  创建Python虚拟环境，安装依赖：
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows下为 .\venv\Scripts\activate
    pip install langchain langchain-ollama langchain-chroma chromadb flask pypdf python-dotenv
    ```
3.  导入知识库：将复习资料放入knowledge目录，运行：
    ```bash
    python ingest_knowledge.py
    ```
4.  启动Web服务：
    ```bash
    python app.py
    ```
5.  浏览器访问 `http://localhost:5000` 即可开始问答。

## 项目亮点
- 完全离线：所有模型和数据均在本地运行，无需联网、无API调用费用，隐私性强。
- 检索精准：基于BGE-M3向量模型做中文文本嵌入，针对408专业知识点优化切分策略，回答内容可溯源到原始文档片段。
- 轻量可扩展：各模块解耦，可轻松替换大模型、向量库，后续可扩展错题本、知识点掌握程度统计等功能。
