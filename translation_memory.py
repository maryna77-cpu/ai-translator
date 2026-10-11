from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path


class TranslationMemory:
    def __init__(self, path: str):
        self.path = Path(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS memory (k TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.conn.commit()

    @staticmethod
    def make_key(payload: dict) -> str:
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, payload: dict):
        key = self.make_key(payload)
        row = self.conn.execute("SELECT value FROM memory WHERE k=?", (key,)).fetchone()
        if not row:
            return None
        return json.loads(row[0])

    def put(self, payload: dict, value):
        key = self.make_key(payload)
        self.conn.execute(
            "INSERT OR REPLACE INTO memory(k,value) VALUES(?,?)",
            (key, json.dumps(value, ensure_ascii=False)),
        )
        self.conn.commit()

    def close(self):
        self.conn.close()
