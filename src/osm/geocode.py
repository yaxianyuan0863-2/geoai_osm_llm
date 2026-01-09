# src/osm/geocode.py
"""
地理编码模块 - 将地名转换为边界框坐标
使用 OpenStreetMap Nominatim API
"""
from __future__ import annotations
from typing import Tuple
import requests
import time


def geocode_to_bbox(
    place: str,
    sleep_s: float = 1.0,
    timeout: int = 30
) -> Tuple[float, float, float, float]:
    """
    将地名转换为边界框坐标
    
    Args:
        place: 地名，如 "Lund", "Malmö", "Stockholm"
        sleep_s: 请求后等待时间（遵守 Nominatim 使用政策）
        timeout: 请求超时时间
    
    Returns:
        (minlon, minlat, maxlon, maxlat) 边界框坐标
    
    Raises:
        ValueError: 如果找不到该地名
        requests.RequestException: 如果请求失败
    """
    url = "https://nominatim.openstreetmap.org/search"
    
    params = {
        "q": place,
        "format": "json",
        "limit": 1,
    }
    
    headers = {
        "User-Agent": "GeoAI-OSM-RAG/1.0 (educational project; contact: student@university.edu)"
    }

    print(f"[Geocode] Looking up: {place}")
    
    response = requests.get(
        url,
        params=params,
        headers=headers,
        timeout=timeout
    )
    response.raise_for_status()
    
    data = response.json()
    
    if not data:
        raise ValueError(f"No geocoding result for place: {place}")

    item = data[0]
    
    # 中心坐标
    lat = float(item["lat"])
    lon = float(item["lon"])
    
    # 边界框: Nominatim 返回 [south, north, west, east]
    bbox = item.get("boundingbox", [])
    
    if len(bbox) >= 4:
        minlat = float(bbox[0])  # south
        maxlat = float(bbox[1])  # north
        minlon = float(bbox[2])  # west
        maxlon = float(bbox[3])  # east
    else:
        # 如果没有边界框，创建一个围绕中心点的小框
        delta = 0.05  # 约 5km
        minlat = lat - delta
        maxlat = lat + delta
        minlon = lon - delta
        maxlon = lon + delta
    
    print(f"[Geocode] Found: center=({lat:.4f}, {lon:.4f}), bbox=({minlon:.4f}, {minlat:.4f}, {maxlon:.4f}, {maxlat:.4f})")
    
    # 遵守 Nominatim 使用政策
    time.sleep(sleep_s)
    
    return (minlon, minlat, maxlon, maxlat)


def geocode_to_center(
    place: str,
    sleep_s: float = 1.0,
    timeout: int = 30
) -> Tuple[float, float]:
    """
    将地名转换为中心坐标
    
    Returns:
        (lon, lat) 中心坐标
    """
    url = "https://nominatim.openstreetmap.org/search"
    
    params = {
        "q": place,
        "format": "json",
        "limit": 1,
    }
    
    headers = {
        "User-Agent": "GeoAI-OSM-RAG/1.0 (educational project)"
    }

    response = requests.get(url, params=params, headers=headers, timeout=timeout)
    response.raise_for_status()
    
    data = response.json()
    
    if not data:
        raise ValueError(f"No geocoding result for place: {place}")

    item = data[0]
    lat = float(item["lat"])
    lon = float(item["lon"])
    
    time.sleep(sleep_s)
    
    return (lon, lat)
