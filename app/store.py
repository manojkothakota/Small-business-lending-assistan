"""SQLite persistence: chunk text + vectors (inspectable, shareable) and a query log for analytics."""
import json
import os
import sqlite3


def connect() -> sqlite3.Connection:
    c = sqlite3.connect(os.getenv("DB_PATH", "rag.db"), check_same_thread=False)
    c.executescript("""CREATE TABLE IF NOT EXISTS chunks(id INTEGER PRIMARY KEY, source TEXT, heading TEXT, text TEXT, vector TEXT);
    CREATE TABLE IF NOT EXISTS query_log(id INTEGER PRIMARY KEY, ts TEXT DEFAULT CURRENT_TIMESTAMP,
                                         question TEXT, grounded INTEGER, cached INTEGER, latency_ms REAL);""")
    return c


def save_chunks(c, chunks, vectors) -> None:
    c.execute("DELETE FROM chunks")
    c.executemany("INSERT INTO chunks VALUES (?,?,?,?,?)", [
        (ch["id"], ch["source"], ch["heading"], ch["text"], json.dumps([round(float(x), 4) for x in vectors[i]]))
        for i, ch in enumerate(chunks)])
    c.commit()


def log_query(c, question, grounded, cached, ms) -> None:
    c.execute("INSERT INTO query_log(question,grounded,cached,latency_ms) VALUES (?,?,?,?)",
              (question, int(grounded), int(cached), ms))
    c.commit()


def count(c, table: str) -> int:
    return c.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
