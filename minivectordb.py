"""
极简版本地向量检索引擎（MiniVectorDB）
完全手写核心逻辑，不依赖Chroma、FAISS等现成向量库
核心功能：文本向量编码、本地持久化、余弦相似度Top-K检索
"""
import json
import numpy as np
import ollama


class MiniVectorDB:
    def __init__(self, model_name: str = "bge-m3", storage_file: str = "vectordb.json"):
        self.model_name = model_name
        self.storage_file = storage_file
        self.documents = []  # 存原始文本
        self.embeddings = []  # 存对应文本的向量
        # 初始化时自动加载历史存储的数据
        self.load()

    def _get_embedding(self, text: str) -> list:
        """调用本地Ollama的bge-m3模型，把文本转成向量"""
        res = ollama.embeddings(model=self.model_name, prompt=text)
        return res["embedding"]

    def _cosine_similarity(self, vec1: list, vec2: list) -> float:
        """手写余弦相似度计算，不用现成库"""
        a = np.array(vec1)
        b = np.array(vec2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def add_texts(self, texts: list):
        """批量添加文本，自动转向量存到内存里"""
        for text in texts:
            if not text.strip():
                continue
            emb = self._get_embedding(text)
            self.documents.append(text)
            self.embeddings.append(emb)
        print(f"✅ 已添加 {len(texts)} 条文本，当前库内共 {len(self.documents)} 条数据")

    def search(self, query: str, top_k: int = 3) -> list:
        """搜索和查询最相似的top_k条文本，返回(文本, 匹配分数)列表"""
        if not self.documents:
            return []
        query_emb = self._get_embedding(query)
        scores = []
        for idx, doc_emb in enumerate(self.embeddings):
            sim = self._cosine_similarity(query_emb, doc_emb)
            scores.append((self.documents[idx], sim))
        # 按相似度从高到低排序，取前top_k个
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def save(self):
        """把当前库内的文本和向量存成本地JSON文件，重启不丢数据"""
        data = {
            "documents": self.documents,
            "embeddings": self.embeddings
        }
        with open(self.storage_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"✅ 已保存到本地文件，共 {len(self.documents)} 条数据")

    def load(self):
        """启动时从本地JSON加载历史数据"""
        import os
        if not os.path.exists(self.storage_file):
            return
        try:
            with open(self.storage_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.documents = data.get("documents", [])
            self.embeddings = data.get("embeddings", [])
            print(f"✅ 从本地加载历史数据，共 {len(self.documents)} 条")
        except Exception as e:
            print(f"加载本地数据失败：{e}")
