#!/usr/bin/env python3

import faiss
import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer

INDEX_DIR = "index"
EMBED_MODEL = "all-MiniLM-L6-v2"

_model = None
_index = None
_chunks = None
_metadata = None

def _load():
    global _model, _index, _chunks, _metadata
    if _index is not None:
        return
    if not os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
        raise FileNotFoundError(
            "No index found."
        )

    _model = SentenceTransformer(EMBED_MODEL)
    _index = faiss.read_index(os.path.join(INDEX_DIR, "index.faiss"))
    with open(os.path.join(INDEX_DIR, "chunks.json")) as f:
        _chunks = json.load(f)
    with open(os.path.join(INDEX_DIR, "metadata.json")) as f:
        _metadata = json.load(f)

def retrieve(query, top_k = 6):
    _load()
    query_vec = _model.encode([query]).astype("float32")
    faiss.normalize_L2(query_vec)

    scores, indices = _index.search(query_vec, top_k)

    results = []
    seen_titles = {} # remove dupes

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        meta = _metadata[idx]
        title = meta["title"]

        seen_titles[title] = seen_titles.get(title, 0) + 1
        if seen_titles[title] > 2:
            continue

        results.append({
            "chunk": _chunks[idx],
            "title": title,
            "authors": meta["authors"],
            "url": meta["url"],
            "arxiv_id": meta["arxiv_id"],
            "score": float(score),
        })

        return results

def get_topic():
    topic_file = os.path.join(INDEX_DIR, "topic.txt")
    if os.path.exists(topic_file):
        with open(topic_file) as f:
            return f.read().strip()
    return "research papers"
