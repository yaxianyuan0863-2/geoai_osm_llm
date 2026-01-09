# src/osm/extractor.py
"""
OSM 数据提取模块
- osmium 命令行工具提取 bbox 子集
- pyosmium 解析并提取特定 tag 的节点
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import subprocess
import shutil

import osmium


@dataclass
class OSMPoint:
    """表示一个 OSM 节点"""
    osm_type: str  # "node"
    osm_id: int
    lon: float
    lat: float
    name: Optional[str]
    tags: Dict[str, str]


class TagNodeHandler(osmium.SimpleHandler):
    """
    pyosmium Handler: 提取匹配特定 tag 的节点
    """
    def __init__(self, key: str, value: str):
        super().__init__()
        self.key = key
        self.value = value
        self.rows: List[OSMPoint] = []

    def node(self, n):
        """处理每个节点"""
        if n.tags.get(self.key) == self.value and n.location.valid():
            self.rows.append(
                OSMPoint(
                    osm_type="node",
                    osm_id=int(n.id),
                    lon=float(n.location.lon),
                    lat=float(n.location.lat),
                    name=n.tags.get("name"),
                    tags=dict(n.tags),
                )
            )


def osmium_extract_bbox(
    input_pbf: Path,
    output_pbf: Path,
    bbox: Tuple[float, float, float, float]
) -> None:
    """
    使用 osmium CLI 工具从大 PBF 文件中提取 bbox 区域
    
    Args:
        input_pbf: 输入的 PBF 文件路径
        output_pbf: 输出的子集 PBF 文件路径
        bbox: (minlon, minlat, maxlon, maxlat) 边界框
    
    Raises:
        FileNotFoundError: 如果找不到输入文件
        RuntimeError: 如果 osmium 执行失败
    """
    # 检查输入文件
    if not input_pbf.exists():
        raise FileNotFoundError(f"Input PBF file not found: {input_pbf}")
    
    # 确保输出目录存在
    output_pbf.parent.mkdir(parents=True, exist_ok=True)
    
    # 构建 bbox 字符串: minlon,minlat,maxlon,maxlat
    bbox_str = ",".join(str(x) for x in bbox)
    
    # 使用 shell=True 方式运行命令（解决 Windows 路径问题）
    cmd = f'osmium extract --bbox {bbox_str} "{input_pbf}" -o "{output_pbf}" -O --set-bounds'
    
    print(f"[osmium] Running: {cmd}")
    
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        error_msg = f"osmium extract failed with exit code {result.returncode}"
        if result.stderr:
            error_msg += f" stderr: {result.stderr}"
        if result.stdout:
            error_msg += f" stdout: {result.stdout}"
        raise RuntimeError(error_msg)
    
    if result.stderr:
        print(f"[osmium] stderr: {result.stderr}")
    
    if not output_pbf.exists():
        raise RuntimeError(f"osmium completed but output file not created: {output_pbf}")
    
    print(f"[osmium] Successfully extracted to {output_pbf}")


def extract_nodes_to_geojson(
    input_pbf: Path,
    key: str,
    value: str,
    out_geojson: Path,
) -> List[OSMPoint]:
    """
    从 PBF 文件中提取匹配 key=value 的节点，保存为 GeoJSON
    
    Args:
        input_pbf: 输入的 PBF 文件
        key: OSM tag 键，如 "amenity"
        value: OSM tag 值，如 "cafe"
        out_geojson: 输出的 GeoJSON 文件路径
    
    Returns:
        提取到的 OSMPoint 列表
    """
    if not input_pbf.exists():
        raise FileNotFoundError(f"Input PBF file not found: {input_pbf}")
    
    print(f"[extractor] Scanning {input_pbf} for {key}={value}")
    
    handler = TagNodeHandler(key, value)
    handler.apply_file(str(input_pbf), locations=True)
    
    print(f"[extractor] Found {len(handler.rows)} nodes")
    
    # 转换为 GeoJSON FeatureCollection
    features = []
    for point in handler.rows:
        feature = {
            "type": "Feature",
            "properties": {
                "osm_type": point.osm_type,
                "osm_id": point.osm_id,
                "name": point.name,
                "tags": point.tags,
            },
            "geometry": {
                "type": "Point",
                "coordinates": [point.lon, point.lat],
            },
        }
        features.append(feature)
    
    feature_collection = {
        "type": "FeatureCollection",
        "features": features,
    }
    
    # 保存 GeoJSON
    out_geojson.parent.mkdir(parents=True, exist_ok=True)
    with open(out_geojson, "w", encoding="utf-8") as f:
        json.dump(feature_collection, f, ensure_ascii=False, indent=2)
    
    print(f"[extractor] Saved GeoJSON to {out_geojson}")
    
    return handler.rows


def extract_ways_to_geojson(
    input_pbf: Path,
    key: str,
    value: str,
    out_geojson: Path,
) -> int:
    """
    提取匹配的 ways（线要素）- 可选扩展
    这里只是一个占位，可以根据需要实现
    """
    # TODO: 实现 way 提取逻辑
    raise NotImplementedError("Way extraction not yet implemented")
