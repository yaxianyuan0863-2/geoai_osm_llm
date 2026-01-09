# app_min.py
"""
Flask Web 服务
提供 API 端点和静态文件服务
"""
from flask import Flask, request, jsonify, send_from_directory, send_file
from pathlib import Path
import traceback

from src.config import OUTPUT_DIR
from src.pipeline import run_query, run_query_without_llm
from src.llm.ollama_client import check_ollama_available, list_available_models

app = Flask(__name__)


@app.get("/")
def home():
    """首页 - 显示 API 信息"""
    return jsonify({
        "service": "GeoAI OSM RAG Demo",
        "endpoints": {
            "POST /chat": "Execute a natural language query",
            "POST /chat_simple": "Execute query without LLM (requires place, key, value)",
            "GET /output/<filename>": "Get output files (e.g., output.geojson)",
            "GET /ui": "Web UI",
            "GET /status": "Check service status",
        }
    })


@app.get("/status")
def status():
    """检查服务状态"""
    ollama_ok = check_ollama_available()
    models = list_available_models() if ollama_ok else []
    
    return jsonify({
        "status": "running",
        "ollama_available": ollama_ok,
        "available_models": models,
    })


@app.post("/chat")
def chat():
    """
    主要 API 端点 - 执行自然语言查询
    
    Request JSON:
        {"query": "Find all cafes in Malmö", "model": "mistral"}
    
    Response JSON:
        {
            "status": "success" | "error",
            "message": "...",
            "geojson_url": "/output/output.geojson",
            "evidence": [...],
            ...
        }
    """
    try:
        data = request.get_json(force=True)
        query = data.get("query", "").strip()
        model = data.get("model", "mistral")
        
        if not query:
            return jsonify({
                "status": "error",
                "message": "Missing 'query' parameter",
            }), 400
        
        print(f"\n{'='*60}")
        print(f"[/chat] Received query: {query}")
        print(f"[/chat] Using model: {model}")
        print('='*60)
        
        result = run_query(query=query, model=model)
        
        if not result.get("success", False):
            return jsonify({
                "status": "error",
                "message": result.get("error", "Unknown error"),
                "query": query,
            }), 500
        
        # 构建成功响应
        message = (
            f"Place: {result['place']}\n"
            f"Chosen tag: {result['chosen_tag']}\n"
            f"Extracted features: {result['count']}\n"
            f"LLM used: {result['llm_ok']}"
        )
        
        if result.get("llm_explanation"):
            message += f"\nLLM explanation: {result['llm_explanation']}"
        
        return jsonify({
            "status": "success",
            "message": message,
            "query": query,
            "place": result["place"],
            "chosen_tag": result["chosen_tag"],
            "count": result["count"],
            "geojson_url": "/output/output.geojson",
            "evidence": result.get("evidence", []),
            "llm_ok": result["llm_ok"],
            "llm_confidence": result.get("llm_confidence", 0),
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": f"Server error: {str(e)}",
            "traceback": traceback.format_exc(),
        }), 500


@app.post("/chat_simple")
def chat_simple():
    """
    简化版 API - 不使用 LLM，直接指定参数
    用于测试或当 LLM 不可用时
    
    Request JSON:
        {
            "query": "...",
            "place": "Lund",
            "key": "amenity",
            "value": "cafe"
        }
    """
    try:
        data = request.get_json(force=True)
        query = data.get("query", "")
        place = data.get("place", "Lund")
        key = data.get("key", "")
        value = data.get("value", "")
        
        if not key or not value:
            return jsonify({
                "status": "error",
                "message": "Missing 'key' or 'value' parameter",
            }), 400
        
        result = run_query_without_llm(query=query, place=place, key=key, value=value)
        
        if not result.get("success", False):
            return jsonify({
                "status": "error",
                "message": result.get("error", "Unknown error"),
            }), 500
        
        return jsonify({
            "status": "success",
            "message": f"Place: {place}\nTag: {key}={value}\nCount: {result['count']}",
            "geojson_url": "/output/output.geojson",
            "count": result["count"],
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": f"Server error: {str(e)}",
        }), 500


@app.get("/output/<path:filename>")
def output_files(filename):
    """提供输出文件（如 GeoJSON）"""
    return send_from_directory(str(OUTPUT_DIR), filename)


@app.get("/ui")
def ui():
    """返回 Web UI"""
    return send_file("chat.html")


@app.errorhandler(404)
def not_found(e):
    return jsonify({"status": "error", "message": "Not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"status": "error", "message": "Internal server error"}), 500


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("GeoAI OSM RAG Demo Server")
    print("="*60)
    
    # 检查 Ollama 状态
    if check_ollama_available():
        models = list_available_models()
        print(f"✓ Ollama is running. Available models: {models}")
    else:
        print("✗ Ollama is not running. Start it with: ollama serve")
        print("  LLM features will not work, but RAG fallback is available.")
    
    print("\nStarting server at http://127.0.0.1:8000")
    print("UI available at http://127.0.0.1:8000/ui")
    print("="*60 + "\n")
    
    app.run(host="127.0.0.1", port=8000, debug=True)
