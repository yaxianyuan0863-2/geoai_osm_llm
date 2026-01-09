# src/rag/wiki_scraper.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import time
import requests
from bs4 import BeautifulSoup

@dataclass
class WikiDoc:
    url: str
    title: str
    text: str

def fetch_wiki_text(url: str, timeout: int = 30) -> WikiDoc:
    headers = {
        "User-Agent": "GeoAI-OSM-RAG/1.0 (educational project)"
    }
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    # Title
    title = soup.title.get_text(strip=True) if soup.title else url

    # Main content: OSM wiki pages usually have content in <div id="content"> or <div id="bodyContent">
    main = soup.find("div", id="content") or soup.find("div", id="bodyContent") or soup.body
    text = main.get_text("\n", strip=True) if main else soup.get_text("\n", strip=True)

    # Light cleanup: remove very short lines
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) >= 3]
    text = "\n".join(lines)

    return WikiDoc(url=url, title=title, text=text)

def save_docs_jsonl(docs: list[WikiDoc], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({"url": d.url, "title": d.title, "text": d.text}, ensure_ascii=False) + "\n")

def scrape_urls(urls: list[str], out_jsonl: Path, sleep_s: float = 0.8) -> None:
    docs: list[WikiDoc] = []
    for i, url in enumerate(urls, 1):
        try:
            doc = fetch_wiki_text(url)
            docs.append(doc)
            print(f"[{i}/{len(urls)}] OK: {url}")
        except Exception as e:
            print(f"[{i}/{len(urls)}] FAIL: {url} -> {e}")
        time.sleep(sleep_s)
    save_docs_jsonl(docs, out_jsonl)
