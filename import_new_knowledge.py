"""
批量导入knowledge目录下所有md文件到向量库
"""
import os
from minivectordb import MiniVectorDB

# 初始化向量库
vdb = MiniVectorDB(storage_file="408_vectordb.json")
print(f"当前向量库已有 {len(vdb.documents)} 条数据")

knowledge_dir = "knowledge"
all_texts = []
for fname in os.listdir(knowledge_dir):
    if fname.endswith(".md"):
        fpath = os.path.join(knowledge_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if len(line) > 10:  # 过滤太短的标题、空行
                    all_texts.append(line)
        print(f"✅ 已读取文件 {fname}")

# 批量添加新文本
vdb.add_texts(all_texts)
vdb.save()
print(f"导入完成！现在向量库共有 {len(vdb.documents)} 条数据")
