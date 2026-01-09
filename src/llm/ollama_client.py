# src/llm/ollama_client.py
"""
Ollama LLM 客户端
用于调用本地 Ollama 服务并解析 JSON 响应
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
import json
import requests

# Ollama 默认配置
OLLAMA_BASE_URL = "http://127.0.0.1:11434"


@dataclass
class LLMResult:
    """LLM 调用结果"""
    ok: bool
    data: Dict[str, Any]
    raw: str


def call_ollama_json(
    model: str,
    system: str,
    user: str,
    timeout_s: int = 120,
    base_url: str = OLLAMA_BASE_URL
) -> LLMResult:
    """
    调用 Ollama chat API 并尝试解析 JSON 响应
    
    Args:
        model: 模型名称，如 "mistral", "llama2", "qwen2"
        system: 系统提示词
        user: 用户输入
        timeout_s: 超时时间（秒）
        base_url: Ollama 服务地址
    
    Returns:
        LLMResult: 包含 ok 状态、解析后的数据和原始响应
    """
    url = f"{base_url}/api/chat"
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,  # 不使用流式响应，等待完整结果
        "options": {
            "temperature": 0.1,  # 低温度以获得更稳定的 JSON 输出
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=timeout_s)
        response.raise_for_status()
        
        result = response.json()
        content = result.get("message", {}).get("content", "")
        
    except requests.exceptions.ConnectionError:
        return LLMResult(
            ok=False,
            data={},
            raw="Error: Cannot connect to Ollama. Is it running? (ollama serve)"
        )
    except requests.exceptions.Timeout:
        return LLMResult(
            ok=False,
            data={},
            raw=f"Error: Ollama request timed out after {timeout_s}s"
        )
    except requests.exceptions.RequestException as e:
        return LLMResult(ok=False, data={}, raw=f"Request error: {str(e)}")
    except Exception as e:
        return LLMResult(ok=False, data={}, raw=f"Unexpected error: {str(e)}")

    # 尝试解析 JSON
    return _parse_json_from_content(content)


def _parse_json_from_content(content: str) -> LLMResult:
    """从 LLM 响应中提取并解析 JSON"""
    
    # 方法 1: 直接解析整个内容
    try:
        data = json.loads(content.strip())
        return LLMResult(ok=True, data=data, raw=content)
    except json.JSONDecodeError:
        pass
    
    # 方法 2: 查找 ```json ... ``` 代码块
    import re
    json_block_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
    if json_block_match:
        try:
            data = json.loads(json_block_match.group(1))
            return LLMResult(ok=True, data=data, raw=content)
        except json.JSONDecodeError:
            pass
    
    # 方法 3: 查找第一个 { 到最后一个 } 之间的内容
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(content[start:end + 1])
            return LLMResult(ok=True, data=data, raw=content)
        except json.JSONDecodeError:
            pass
    
    # 无法解析
    return LLMResult(ok=False, data={}, raw=content)


def check_ollama_available(base_url: str = OLLAMA_BASE_URL) -> bool:
    """检查 Ollama 服务是否可用"""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def list_available_models(base_url: str = OLLAMA_BASE_URL) -> list:
    """列出可用的模型"""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
    except:
        pass
    return []
