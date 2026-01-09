#!/usr/bin/env python3
# test_llm_integration.py
"""
测试 LLM 集成是否正常工作
运行此脚本检查：
1. Ollama 服务是否运行
2. LLM 能否正确解析查询
3. 完整 pipeline 是否工作
"""
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))


def test_ollama_connection():
    """测试 Ollama 连接"""
    print("\n" + "="*60)
    print("Test 1: Ollama Connection")
    print("="*60)
    
    from src.llm.ollama_client import check_ollama_available, list_available_models
    
    available = check_ollama_available()
    print(f"Ollama available: {available}")
    
    if available:
        models = list_available_models()
        print(f"Available models: {models}")
        return True
    else:
        print("ERROR: Ollama is not running!")
        print("Start it with: ollama serve")
        print("Then pull a model: ollama pull mistral")
        return False


def test_llm_json_parsing():
    """测试 LLM JSON 解析"""
    print("\n" + "="*60)
    print("Test 2: LLM JSON Parsing")
    print("="*60)
    
    from src.llm.ollama_client import call_ollama_json
    
    system = "You are a helpful assistant. Return only valid JSON."
    user = 'Return this JSON: {"name": "test", "value": 42}'
    
    result = call_ollama_json(model="mistral", system=system, user=user)
    
    print(f"Result OK: {result.ok}")
    print(f"Data: {result.data}")
    print(f"Raw (first 200 chars): {result.raw[:200]}")
    
    return result.ok


def test_rag_retrieval():
    """测试 RAG 检索"""
    print("\n" + "="*60)
    print("Test 3: RAG Retrieval")
    print("="*60)
    
    try:
        from src.rag.retriever import FaissRetriever, pick_tag_from_chunks
        
        retriever = FaissRetriever()
        query = "Find all cafes"
        chunks = retriever.retrieve(query, k=3)
        
        print(f"Query: {query}")
        print(f"Retrieved {len(chunks)} chunks:")
        
        for i, chunk in enumerate(chunks, 1):
            print(f"\n  [{i}] Score: {chunk.score:.3f}")
            print(f"      Tag: {chunk.key}={chunk.value}")
            print(f"      URL: {chunk.url}")
        
        key, value = pick_tag_from_chunks(chunks)
        print(f"\nSelected tag: {key}={value}")
        
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def test_llm_query_parsing():
    """测试 LLM 查询解析"""
    print("\n" + "="*60)
    print("Test 4: LLM Query Parsing")
    print("="*60)
    
    try:
        from src.rag.retriever import FaissRetriever
        from src.query.llm_parser import llm_parse_query
        
        retriever = FaissRetriever()
        query = "Find all cafes in Malmö"
        chunks = retriever.retrieve(query, k=5)
        
        print(f"Query: {query}")
        print("Calling LLM...")
        
        result = llm_parse_query(query=query, chunks=chunks, model="mistral")
        
        print(f"\nLLM Result OK: {result['ok']}")
        if result['ok']:
            data = result['data']
            print(f"  Place: {data.get('place')}")
            print(f"  Tag: {data.get('tag')}")
            print(f"  Confidence: {data.get('confidence')}")
            print(f"  Explanation: {data.get('explanation')}")
        else:
            print(f"  Raw output: {result['raw'][:500]}")
        
        return result['ok']
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_geocoding():
    """测试地理编码"""
    print("\n" + "="*60)
    print("Test 5: Geocoding")
    print("="*60)
    
    try:
        from src.osm.geocode import geocode_to_bbox
        
        place = "Lund"
        bbox = geocode_to_bbox(place)
        
        print(f"Place: {place}")
        print(f"BBox: {bbox}")
        print(f"  minlon={bbox[0]:.4f}, minlat={bbox[1]:.4f}")
        print(f"  maxlon={bbox[2]:.4f}, maxlat={bbox[3]:.4f}")
        
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def main():
    print("\n" + "="*60)
    print("GeoAI LLM Integration Test Suite")
    print("="*60)
    
    results = {}
    
    # Test 1: Ollama
    results['ollama'] = test_ollama_connection()
    
    if not results['ollama']:
        print("\n⚠️  Ollama not available. Skipping LLM tests.")
        print("   Start Ollama: ollama serve")
        print("   Pull model: ollama pull mistral")
    else:
        # Test 2: LLM JSON
        results['llm_json'] = test_llm_json_parsing()
    
    # Test 3: RAG
    results['rag'] = test_rag_retrieval()
    
    if results.get('ollama') and results.get('rag'):
        # Test 4: LLM Query Parsing
        results['llm_query'] = test_llm_query_parsing()
    
    # Test 5: Geocoding
    results['geocoding'] = test_geocoding()
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(results.values())
    print("\n" + ("All tests passed! ✓" if all_passed else "Some tests failed ✗"))
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
