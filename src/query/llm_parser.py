# src/query/llm_parser.py
"""
LLM 查询解析器
将用户自然语言查询 + RAG 证据 -> 结构化 JSON (place + tag)
"""
from __future__ import annotations
from typing import Dict, Any, List
from src.llm.ollama_client import call_ollama_json

# 关键 System Prompt - 这是让 LLM 正确输出的核心
SYSTEM_PROMPT = """You are a careful GIS assistant for OpenStreetMap.
You MUST base decisions on the provided evidence (OSM Wiki snippets).
Return ONLY valid JSON. No extra text, no markdown, no explanation outside JSON.

Required JSON Schema:
{
  "place": "<string or null>",
  "tag": {"key": "<string>", "value": "<string>"},
  "confidence": 0.0-1.0,
  "explanation": "<short reason for your choice>"
}

Rules:
1. If the user query mentions a place (city, region, area), extract it to "place".
   If no place is mentioned, set "place" to null.
2. Choose tag.key and tag.value based on the evidence snippets.
   It must be a valid OSM tag pair like amenity=cafe, highway=bus_stop, shop=supermarket.
3. Only pick tags that appear in the evidence. Do not invent tags.
4. Set confidence based on how well the evidence matches the query.

Examples:
- Query "Find all cafes in Malmö" -> {"place": "Malmö", "tag": {"key": "amenity", "value": "cafe"}, ...}
- Query "Show bus stops" -> {"place": null, "tag": {"key": "highway", "value": "bus_stop"}, ...}
- Query "restaurants in Lund" -> {"place": "Lund", "tag": {"key": "amenity", "value": "restaurant"}, ...}
"""


def format_evidence(chunks: List) -> str:
    """
    将 RAG 检索到的 chunks 格式化为 LLM 可读的证据文本
    
    Args:
        chunks: RetrievedChunk 列表
    
    Returns:
        格式化的证据字符串
    """
    lines = []
    for i, c in enumerate(chunks, 1):
        # 截取 snippet，避免太长
        snippet = c.page_content[:600].replace("\n", " ").strip()
        lines.append(
            f"[Evidence {i}]\n"
            f"  Tag: {c.key}={c.value}\n"
            f"  URL: {c.url}\n"
            f"  Score: {c.score:.3f}\n"
            f"  Content: {snippet}\n"
        )
    return "\n".join(lines)


def llm_parse_query(
    query: str,
    chunks: List,
    model: str = "mistral",
) -> Dict[str, Any]:
    """
    使用 LLM 解析用户查询
    
    Args:
        query: 用户的自然语言查询
        chunks: RAG 检索到的证据 chunks
        model: Ollama 模型名称
    
    Returns:
        {
            "ok": bool,
            "data": {"place": str|None, "tag": {"key": str, "value": str}, ...},
            "raw": str  # LLM 原始输出
        }
    """
    evidence = format_evidence(chunks)
    
    user_prompt = f"""User Query:
"{query}"

Evidence from OSM Wiki (retrieved by semantic search):
{evidence}

Based on the user query and evidence above, return a JSON object with:
- "place": the location mentioned in the query (or null if none)
- "tag": the best matching OSM tag from the evidence {{"key": "...", "value": "..."}}
- "confidence": your confidence score 0.0-1.0
- "explanation": brief reason for your choice

Return ONLY the JSON object, nothing else."""

    result = call_ollama_json(model=model, system=SYSTEM_PROMPT, user=user_prompt)
    
    if result.ok:
        return {"ok": True, "data": result.data, "raw": result.raw}
    return {"ok": False, "data": {}, "raw": result.raw}


def validate_llm_response(data: Dict) -> bool:
    """验证 LLM 响应是否符合预期格式"""
    if not isinstance(data, dict):
        return False
    
    tag = data.get("tag")
    if not isinstance(tag, dict):
        return False
    
    key = tag.get("key")
    value = tag.get("value")
    
    if not key or not value:
        return False
    
    if not isinstance(key, str) or not isinstance(value, str):
        return False
    
    return True
