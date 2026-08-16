"""File-backed, thread-safe cache for AI report artifacts.

Each match_id maps to one JSON file under ``directory``. Values are dicts with
a ``saved_at`` epoch timestamp; ``get`` enforces a TTL and promotes disk hits
into the in-memory layer. Writes are atomic (tmp + rename) so a crash never
corrupts a report. Survives server restarts and is shared across processes,
which is what makes pre-warmed reports fast for every request.
"""

import json
import threading
import time
from pathlib import Path


class ReportCache:
    TTL = 6 * 3600  # reports are regenerated ~4x per match per day

    def __init__(self, directory="data/cache/reports"):
        self.directory = Path(directory)
        self._mem = {}
        self._lock = threading.Lock()

    def _path(self, match_id):
        return self.directory / f"{match_id}.json"

    def get(self, match_id):
        match_id = str(match_id)
        with self._lock:
            entry = self._mem.get(match_id)
            if entry and time.time() - entry[0] < self.TTL:
                return entry[1]
        try:
            path = self._path(match_id)
            payload = json.loads(path.read_text(encoding="utf-8"))
            if time.time() - payload.get("saved_at", 0) < self.TTL:
                with self._lock:
                    self._mem[match_id] = (payload["saved_at"], payload)
                return payload
        except (OSError, ValueError, TypeError):
            pass
        return None

    def set(self, match_id, payload):
        match_id = str(match_id)
        now = time.time()
        payload = dict(payload)
        payload["saved_at"] = now
        with self._lock:
            self._mem[match_id] = (now, payload)
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self._path(match_id)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            temporary.replace(path)
        except OSError as exc:
            print(f"ReportCache write failed for {match_id}: {exc}")