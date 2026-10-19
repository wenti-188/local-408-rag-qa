"""
对比测试：验证手写向量库的检索准确率
"""
from minivectordb import MiniVectorDB

db = MiniVectorDB(storage_file="408_vectordb.json")
print(f"当前知识库共 {len(db.documents)} 个片段")

test_questions = [
    "TCP三次握手的过程是什么",
    "进程调度有哪些算法",
    "栈的特点是什么",
    "什么是虚拟内存",
    "CSMA/CD协议的工作原理"
]

print("\n===== 手写向量库检索测试结果 =====")
for q in test_questions:
    results = db.search(q, top_k=1)
    doc, score = results[0]
    print(f"\n问题：{q}")
    print(f"匹配到：{doc[:80]}...")
    print(f"相似度：{score:.3f}")
