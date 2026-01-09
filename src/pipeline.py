# src/pipeline.py
"""
核心 Pipeline - 整合 RAG → LLM → Geocode → OSM Extract → GeoJSON
"""
from pathlib import Path
from typing import Dict, Any, Optional
import re
import unicodedata

from src.config import OSM_PBF, OUTPUT_DIR, OUTPUT_GEOJSON
from src.rag.retriever import FaissRetriever, pick_tag_from_chunks
from src.osm.extractor import extract_nodes_to_geojson, osmium_extract_bbox
from src.osm.geocode import geocode_to_bbox
from src.query.llm_parser import llm_parse_query, validate_llm_response

DEFAULT_PLACE = "Lund"


def safe_slug(text: str) -> str:
    """
    将文本转换为安全的文件名 slug
    例如: "Malmö" -> "malmo", "New York" -> "new_york"
    """
    # 转小写
    text = text.lower()
    # 规范化 Unicode (NFD 分解后移除变音符号)
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    # 替换空格和特殊字符
    text = re.sub(r'[^a-z0-9]+', '_', text)
    # 移除首尾下划线
    text = text.strip('_')
    return text or "unknown"


def simple_place_heuristic(query: str) -> Optional[str]:
    """
    简单的地点提取规则
    匹配 "in <Place>" 模式
    """
    patterns = [
        r"\bin\s+([A-Za-zÅÄÖåäöØøÆæ\-\s]{2,})$",  # "in Malmö"
        r"\bin\s+([A-Za-zÅÄÖåäöØøÆæ\-\s]{2,})\s*$",
        r"(?:from|at|near)\s+([A-Za-zÅÄÖåäöØøÆæ\-\s]{2,})$",
    ]
    
    query_clean = query.strip()
    for pattern in patterns:
        m = re.search(pattern, query_clean, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


def run_query(query: str, model: str = "mistral") -> Dict[str, Any]:
    """
    执行完整的查询流程
    
    Args:
        query: 用户的自然语言查询，如 "Find all cafes in Malmö"
        model: Ollama 模型名称
    
    Returns:
        包含查询结果的字典
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # ========== Step 1: RAG 检索 ==========
    print(f"[Pipeline] Step 1: RAG retrieval for query: {query}")
    try:
        retriever = FaissRetriever()
        chunks = retriever.retrieve(query, k=5)
        print(f"[Pipeline] Retrieved {len(chunks)} chunks")
    except Exception as e:
        return {
            "query": query,
            "error": f"RAG retrieval failed: {str(e)}",
            "success": False
        }
    
    # ========== Step 2: LLM 解析 ==========
    print(f"[Pipeline] Step 2: LLM parsing with model={model}")
    llm_res = llm_parse_query(query=query, chunks=chunks, model=model)
    llm_ok = llm_res.get("ok", False) and validate_llm_response(llm_res.get("data", {}))
    print(f"[Pipeline] LLM result ok={llm_ok}")
    
    # ========== Step 3: 决定 Place ==========
    place = None
    if llm_ok:
        place = llm_res["data"].get("place")
        if place:
            print(f"[Pipeline] Place from LLM: {place}")
    
    if not place:
        place = simple_place_heuristic(query)
        if place:
            print(f"[Pipeline] Place from heuristic: {place}")
    
    if not place:
        place = DEFAULT_PLACE
        print(f"[Pipeline] Using default place: {place}")
    
    # ========== Step 4: 决定 Tag ==========
    key = value = None
    if llm_ok:
        tag_data = llm_res["data"].get("tag", {})
        key = tag_data.get("key")
        value = tag_data.get("value")
        if key and value:
            print(f"[Pipeline] Tag from LLM: {key}={value}")
    
    if not key or not value:
        # Fallback: 从 RAG chunks 投票选择
        print("[Pipeline] Falling back to RAG voting for tag")
        try:
            key, value = pick_tag_from_chunks(chunks)
            print(f"[Pipeline] Tag from RAG voting: {key}={value}")
        except ValueError as e:
            return {
                "query": query,
                "error": f"Could not determine OSM tag: {str(e)}",
                "success": False
            }
    
    # ========== Step 5: Geocode -> BBox ==========
    print(f"[Pipeline] Step 5: Geocoding {place}")
    try:
        bbox = geocode_to_bbox(place)
        print(f"[Pipeline] BBox: {bbox}")
    except Exception as e:
        return {
            "query": query,
            "place": place,
            "error": f"Geocoding failed for '{place}': {str(e)}",
            "success": False
        }
    
    # ========== Step 6: OSM Extract (bbox -> sub pbf) ==========
    place_slug = safe_slug(place)
    sub_pbf = OUTPUT_DIR / f"sub_{place_slug}.osm.pbf"
    
    print(f"[Pipeline] Step 6: Extracting bbox subset to {sub_pbf}")
    try:
        osmium_extract_bbox(Path(OSM_PBF), sub_pbf, bbox)
    except Exception as e:
        return {
            "query": query,
            "place": place,
            "chosen_tag": f"{key}={value}",
            "error": f"OSM extraction failed: {str(e)}",
            "success": False
        }
    
    # ========== Step 7: Extract nodes -> GeoJSON ==========
    print(f"[Pipeline] Step 7: Extracting nodes with {key}={value}")
    try:
        rows = extract_nodes_to_geojson(sub_pbf, key, value, Path(OUTPUT_GEOJSON))
        print(f"[Pipeline] Extracted {len(rows)} nodes")
    except Exception as e:
        return {
            "query": query,
            "place": place,
            "chosen_tag": f"{key}={value}",
            "error": f"Node extraction failed: {str(e)}",
            "success": False
        }
    
    # ========== 构建返回结果 ==========
    evidence = [
        {
            "score": round(c.score, 4),
            "key": c.key,
            "value": c.value,
            "url": c.url,
            "snippet": c.page_content[:220].replace("\n", " "),
        }
        for c in chunks
    ]
    
    return {
        "success": True,
        "query": query,
        "place": place,
        "chosen_tag": f"{key}={value}",
        "bbox": bbox,
        "count": len(rows),
        "geojson_path": str(OUTPUT_GEOJSON),
        "evidence": evidence,
        "llm_ok": llm_ok,
        "llm_raw": llm_res.get("raw", ""),
        "llm_explanation": llm_res.get("data", {}).get("explanation", ""),
        "llm_confidence": llm_res.get("data", {}).get("confidence", 0),
    }


def run_query_without_llm(query: str, place: str, key: str, value: str) -> Dict[str, Any]:
    """
    不使用 LLM 的简化查询（用于测试或 fallback）
    直接使用指定的 place, key, value
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Geocode
    try:
        bbox = geocode_to_bbox(place)
    except Exception as e:
        return {"success": False, "error": f"Geocoding failed: {str(e)}"}
    
    # Extract
    place_slug = safe_slug(place)
    sub_pbf = OUTPUT_DIR / f"sub_{place_slug}.osm.pbf"
    
    try:
        osmium_extract_bbox(Path(OSM_PBF), sub_pbf, bbox)
    except Exception as e:
        return {"success": False, "error": f"OSM extraction failed: {str(e)}"}
    
    # Nodes to GeoJSON
    try:
        rows = extract_nodes_to_geojson(sub_pbf, key, value, Path(OUTPUT_GEOJSON))
    except Exception as e:
        return {"success": False, "error": f"Node extraction failed: {str(e)}"}
    
    return {
        "success": True,
        "query": query,
        "place": place,
        "chosen_tag": f"{key}={value}",
        "bbox": bbox,
        "count": len(rows),
        "geojson_path": str(OUTPUT_GEOJSON),
        "llm_ok": False,
    }
