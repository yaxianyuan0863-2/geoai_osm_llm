# src/rag/retriever.py
"""
RAG 检索器 - 使用 FAISS 进行语义搜索
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import FAISS_INDEX, FAISS_META


@dataclass
class RetrievedChunk:
    """检索到的文档块"""
    score: float          # 相似度分数
    page_content: str     # 文档内容
    url: str              # 来源 URL
    title: str            # 页面标题
    key: str | None       # OSM tag key (如 "amenity")
    value: str | None     # OSM tag value (如 "cafe")


class FaissRetriever:
    """
    FAISS 向量检索器
    从预建的索引中检索与查询最相似的文档块
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_path=FAISS_INDEX,
        meta_path=FAISS_META,
    ):
        """
        初始化检索器
        
        Args:
            model_name: Sentence Transformer 模型名称
            index_path: FAISS 索引文件路径
            meta_path: 元数据 JSON 文件路径
        """
        print(f"[Retriever] Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        print(f"[Retriever] Loading FAISS index: {index_path}")
        self.index = faiss.read_index(str(index_path))
        
        print(f"[Retriever] Loading metadata: {meta_path}")
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta: List[Dict] = json.load(f)
        
        print(f"[Retriever] Loaded {len(self.meta)} chunks")

    def retrieve(self, query: str, k: int = 5) -> List[RetrievedChunk]:
        """
        检索与查询最相似的 k 个文档块
        
        Args:
            query: 用户查询文本
            k: 返回的结果数量
        
        Returns:
            RetrievedChunk 列表，按相似度降序排列
        """
        # 将查询编码为向量
        query_vector = self.model.encode(
            query,
            normalize_embeddings=True
        ).astype(np.float32)[None, :]
        
        # FAISS 搜索
        distances, indices = self.index.search(query_vector, k)
        
        # 构建结果
        results: List[RetrievedChunk] = []
        for rank, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.meta):
                continue
            
            meta = self.meta[int(idx)]
            results.append(
                RetrievedChunk(
                    score=float(distances[0][rank]),
                    page_content=meta.get("page_content", ""),
                    url=meta.get("url", ""),
                    title=meta.get("title", ""),
                    key=meta.get("key"),
                    value=meta.get("value"),
                )
            )
        
        return results


def pick_tag_from_chunks(chunks: List[RetrievedChunk]) -> Tuple[str, str]:
    """
    从检索结果中投票选择最可能的 OSM tag
    
    使用加权投票：相似度分数作为权重
    
    Args:
        chunks: 检索到的文档块列表
    
    Returns:
        (key, value) 元组
    
    Raises:
        ValueError: 如果没有找到有效的 tag
    """
    votes: Dict[Tuple[str, str], float] = {}
    
    for chunk in chunks:
        if chunk.key and chunk.value:
            tag = (chunk.key, chunk.value)
            votes[tag] = votes.get(tag, 0.0) + chunk.score
    
    if not votes:
        raise ValueError(
            "No valid (key, value) found in retrieved chunks. "
            "Consider expanding wiki seeds or adjusting the query."
        )
    
    # 选择得分最高的 tag
    best_tag = max(votes.items(), key=lambda x: x[1])[0]
    return best_tag[0], best_tag[1]
