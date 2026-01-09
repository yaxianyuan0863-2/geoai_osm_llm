# src/config.py
"""
项目配置文件 - 统一管理所有路径常量
"""
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).resolve().parents[1]

# 数据目录
DATA_DIR = ROOT / "data"

# OSM 数据文件路径 (需要从 Geofabrik 下载)
# 示例: sweden-latest.osm.pbf
OSM_PBF = DATA_DIR / "osm" / "sweden-251214.osm.pbf"

# Wiki 抓取输出目录
WIKI_RAW_DIR = DATA_DIR / "wiki_raw"
WIKI_CHUNKS_DIR = DATA_DIR / "wiki_chunks"

# FAISS 索引目录
FAISS_DIR = ROOT / "faiss_index"
FAISS_INDEX = FAISS_DIR / "faiss_index"
FAISS_META = FAISS_DIR / "faiss_index.metadata.json"

# 输出目录
OUTPUT_DIR = ROOT / "output"
OUTPUT_GEOJSON = OUTPUT_DIR / "output.geojson"
