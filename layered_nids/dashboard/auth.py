"""NIDS dashboard sign-in: PBKDF2 password hashes, random session tokens, throttling of failed attempts.

Users live in local/users.json (never committed): {name: {"salt": hex, "iterations": n, "hash": hex}}. `create_user` makes a random
password, stores only its hash, and writes the password itself to local/PASSWORD.txt for the owner to read once.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import threading
import time
from pathlib import Path

ITERATIONS = 240_000
SESSION_SECONDS = 8 * 3600
MAX_FAILS, LOCK_SECONDS = 5, 60


def hash_password(password: str, salt: bytes | None = None, iterations: int = ITERATIONS) -> dict:
    salt = salt or secrets.token_bytes(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return {"salt": salt.hex(), "iterations": iterations, "hash": h.hex()}


def verify_password(password: str, record: dict) -> bool:
    h = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(record["salt"]), int(record["iterations"]))
    return hmac.compare_digest(h.hex(), record["hash"])


def create_user(users_file: Path, name: str, password_file: Path) -> None:
    users = json.loads(users_file.read_text()) if users_file.exists() else {}
    password = secrets.token_urlsafe(18)
    users[name] = hash_password(password)
    users_file.parent.mkdir(parents=True, exist_ok=True)
    users_file.write_text(json.dumps(users))
    password_file.write_text(f"user: {name}\npassword: {password}\n")


class Sessions:
    def __init__(self, users_file: Path):
        self.users_file = Path(users_file)
        self._s: dict = {}
        self._fails: dict = {}
        self._lock = threading.Lock()

    def _users(self) -> dict:
        return json.loads(self.users_file.read_text()) if self.users_file.exists() else {}

    def login(self, name: str, password: str, client: str):
        """Returns a session token or None. Unknown user and wrong password take the same time and give the same answer."""
        now = time.time()
        with self._lock:
            count, since = self._fails.get(client, (0, now))
            if count >= MAX_FAILS and now - since < LOCK_SECONDS:
                return None
        users = self._users()
        rec = users.get(name) or hash_password("x", b"\x00" * 16)      # dummy record: same work for unknown users
        ok = verify_password(password, rec) and name in users
        with self._lock:
            if not ok:
                count, since = self._fails.get(client, (0, now))
                self._fails[client] = (count + 1, since if count else now)
                return None
            self._fails.pop(client, None)
            token = secrets.token_urlsafe(32)
            self._s[token] = (name, now + SESSION_SECONDS)
            return token

    def user(self, token: str | None):
        if not token:
            return None
        with self._lock:
            rec = self._s.get(token)
            if rec and rec[1] > time.time():
                return rec[0]
            self._s.pop(token, None)
            return None

    def logout(self, token: str | None) -> None:
        with self._lock:
            self._s.pop(token, None)
