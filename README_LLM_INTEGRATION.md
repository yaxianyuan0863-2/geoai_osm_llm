# GeoAI OSM LLM Project - LLM Integration Guide

## 项目概述

这是一个使用 LLM 和 RAG 技术从 OpenStreetMap 数据中提取地理要素的系统。

### 工作流程

```
用户查询 (自然语言)
    ↓
RAG 检索 (从 OSM Wiki 知识库中找相关 tag 定义)
    ↓
LLM 解析 (理解查询，选择最佳 tag，提取地点)
    ↓
地理编码 (地名 → 边界框坐标)
    ↓
OSM 提取 (从大 PBF 文件裁剪子集)
    ↓
节点抽取 (按 tag 过滤节点)
    ↓
GeoJSON 输出 (可视化在地图上)
```

## 环境设置

### 1. 激活 Conda 环境

```bash
conda activate geoai_project_env
```

### 2. 安装 osmium-tool

```bash
conda install -c conda-forge osmium-tool
```

### 3. 启动 Ollama 服务

```bash
# 在一个终端中启动 Ollama 服务
ollama serve

# 在另一个终端中拉取模型
ollama pull mistral
# 或者其他模型
ollama pull qwen2
ollama pull llama3
```

### 4. 验证 Ollama 运行状态

```bash
# 检查服务是否运行
curl http://127.0.0.1:11434/api/tags

# 测试生成
ollama run mistral "Hello"
```

## 运行测试

### 测试 LLM 集成

```bash
cd /path/to/project
python test_llm_integration.py
```

这会测试：
1. Ollama 连接
2. LLM JSON 解析
3. RAG 检索
4. LLM 查询解析
5. 地理编码

### 启动 Web 服务

```bash
python app_min.py
```

然后访问：
- Web UI: http://127.0.0.1:8000/ui
- API: http://127.0.0.1:8000/chat

## API 使用

### POST /chat - 自然语言查询

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Find all cafes in Malmö", "model": "mistral"}'
```

响应示例：
```json
{
  "status": "success",
  "message": "Place: Malmö\nChosen tag: amenity=cafe\nExtracted features: 127\nLLM used: true",
  "place": "Malmö",
  "chosen_tag": "amenity=cafe",
  "count": 127,
  "geojson_url": "/output/output.geojson",
  "evidence": [...],
  "llm_ok": true,
  "llm_confidence": 0.95
}
```

### POST /chat_simple - 不使用 LLM 的简化查询

```bash
curl -X POST http://127.0.0.1:8000/chat_simple \
  -H "Content-Type: application/json" \
  -d '{"place": "Lund", "key": "amenity", "value": "cafe"}'
```

### GET /status - 检查服务状态

```bash
curl http://127.0.0.1:8000/status
```

## 关键文件说明

| 文件 | 作用 |
|------|------|
| `src/llm/ollama_client.py` | Ollama API 客户端，调用 LLM 并解析 JSON |
| `src/query/llm_parser.py` | LLM 查询解析器，核心 Prompt 在这里 |
| `src/rag/retriever.py` | FAISS 向量检索器 |
| `src/osm/extractor.py` | OSM 数据提取（osmium + pyosmium） |
| `src/osm/geocode.py` | Nominatim 地理编码 |
| `src/pipeline.py` | 主 Pipeline，串联所有模块 |
| `app_min.py` | Flask Web 服务 |
| `chat.html` | 前端 UI |

## 核心 Prompt

LLM 解析使用的 System Prompt（在 `src/query/llm_parser.py` 中）：

```
You are a careful GIS assistant for OpenStreetMap.
You MUST base decisions on the provided evidence (OSM Wiki snippets).
Return ONLY valid JSON. No extra text.

Schema:
{
  "place": "<string or null>",
  "tag": {"key": "<string>", "value": "<string>"},
  "confidence": 0.0-1.0,
  "explanation": "<short>"
}

Rules:
- If place is not explicitly mentioned, set place to null.
- tag.key and tag.value must be a single OSM tag pair like amenity=cafe.
```

## 故障排除

### Ollama 连接失败

```bash
# 检查 Ollama 是否运行
ps aux | grep ollama

# 重启 Ollama
pkill ollama
ollama serve
```

### osmium 命令找不到

```bash
# 确保在正确的 conda 环境中
conda activate geoai_project_env

# 重新安装
conda install -c conda-forge osmium-tool

# 验证
which osmium
```

### FAISS 索引不存在

需要先运行知识库构建脚本：

```bash
python src/step2_scrape_wiki.py  # 抓取 wiki
python src/step3_build_faiss.py   # 构建 FAISS 索引
```

### LLM 返回无效 JSON

1. 检查 Ollama 日志
2. 尝试更大的模型（如 llama3 或 qwen2）
3. 检查 `llm_parser.py` 中的 prompt

## 下一步开发建议

1. **扩展 Wiki 种子 URL** - 添加更多 tag 页面到 `seed_urls.py`
2. **支持 Ways 和 Relations** - 目前只支持 nodes
3. **缓存机制** - 缓存 geocoding 和 LLM 结果
4. **更好的错误处理** - 添加重试逻辑
5. **QGIS 插件** - 根据 instruction 的要求开发 QGIS 插件
