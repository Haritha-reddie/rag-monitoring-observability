"""
src/monitoring/cache.py
------------------------
Query cache for high-frequency enterprise queries.

Caches:
- Query → response mapping
- TTL-based expiry (default 1 hour)
- LRU eviction when cache is full

Why this matters:
  Repeated queries (e.g. "What is the return policy?" asked 100x/day)
  should not hit the LLM every time.
  Cache reduces:
    - Latency: from ~3s to ~5ms
    - Cost: from $0.001/query to $0
    - Load: on Ollama/Groq API

Production systems use Redis. This uses a local JSON file
as a lightweight equivalent that demonstrates the same concept.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, asdict

CACHE_FILE  = Path("query_cache.json")
MAX_ENTRIES = 500       # LRU eviction after this many entries
DEFAULT_TTL = 3600      # 1 hour in seconds


@dataclass
class CacheEntry:
    query:      str
    response:   dict
    created_at: float
    expires_at: float
    hit_count:  int = 0


class QueryCache:
    """
    TTL-based query cache with LRU eviction.

    Usage:
        cache = QueryCache()

        # Check cache before running RAG
        cached = cache.get("What is the return policy?")
        if cached:
            return cached  # instant response

        # Run RAG pipeline
        response = rag_chain.run(query)

        # Store in cache
        cache.set("What is the return policy?", response)
    """

    def __init__(
        self,
        cache_file: Path = CACHE_FILE,
        max_entries: int = MAX_ENTRIES,
        default_ttl: int = DEFAULT_TTL,
    ):
        self.cache_file  = cache_file
        self.max_entries = max_entries
        self.default_ttl = default_ttl
        self._cache: dict[str, dict] = {}
        self._load()

    # ── Key generation ─────────────────────────────────────────

    def _key(self, query: str) -> str:
        """Generate a stable cache key from the query."""
        normalized = query.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    # ── Public API ─────────────────────────────────────────────

    def get(self, query: str) -> Optional[dict]:
        """
        Retrieve a cached response.
        Returns None if not found or expired.
        Updates hit count on successful retrieval.
        """
        key   = self._key(query)
        entry = self._cache.get(key)

        if entry is None:
            return None

        # Check expiry
        if time.time() > entry["expires_at"]:
            del self._cache[key]
            self._save()
            return None

        # Update hit count
        entry["hit_count"] += 1
        self._save()

        print(f"  💾 Cache HIT (hits={entry['hit_count']}) — {query[:50]}")
        return entry["response"]

    def set(
        self,
        query:    str,
        response: dict,
        ttl:      int = None,
    ) -> None:
        """
        Store a response in the cache.
        Evicts LRU entries if cache is full.
        """
        if ttl is None:
            ttl = self.default_ttl

        key = self._key(query)
        now = time.time()

        self._cache[key] = {
            "query":      query,
            "response":   response,
            "created_at": now,
            "expires_at": now + ttl,
            "hit_count":  0,
        }

        # LRU eviction if over max size
        if len(self._cache) > self.max_entries:
            self._evict_lru()

        self._save()
        print(f"  💾 Cache SET (ttl={ttl}s) — {query[:50]}")

    def invalidate(self, query: str) -> bool:
        """Remove a specific query from the cache."""
        key = self._key(query)
        if key in self._cache:
            del self._cache[key]
            self._save()
            return True
        return False

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache = {}
        self._save()

    def stats(self) -> dict:
        """Return cache statistics."""
        now     = time.time()
        active  = [e for e in self._cache.values() if e["expires_at"] > now]
        expired = len(self._cache) - len(active)
        total_hits = sum(e["hit_count"] for e in active)

        return {
            "total_entries":   len(self._cache),
            "active_entries":  len(active),
            "expired_entries": expired,
            "total_hits":      total_hits,
            "max_entries":     self.max_entries,
        }

    # ── Internal ───────────────────────────────────────────────

    def _evict_lru(self) -> None:
        """Remove least recently used entries until under max_entries."""
        # Sort by created_at ascending (oldest first)
        sorted_keys = sorted(
            self._cache.keys(),
            key=lambda k: self._cache[k]["created_at"]
        )
        # Remove oldest 10%
        to_remove = max(1, len(sorted_keys) // 10)
        for key in sorted_keys[:to_remove]:
            del self._cache[key]

    def _load(self) -> None:
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    self._cache = json.load(f)
                # Prune expired entries on load
                now = time.time()
                self._cache = {
                    k: v for k, v in self._cache.items()
                    if v.get("expires_at", 0) > now
                }
            except Exception:
                self._cache = {}

    def _save(self) -> None:
        """Persist cache to disk."""
        try:
            with open(self.cache_file, "w") as f:
                json.dump(self._cache, f, indent=2)
        except Exception:
            pass  # Cache write failure is non-fatal


# Global singleton
_cache: Optional[QueryCache] = None

def get_cache() -> QueryCache:
    global _cache
    if _cache is None:
        _cache = QueryCache()
    return _cache
