"""A small snapshot on disk, and an HTTP server that serves it like the origin.

`build(root, sid)` writes `<root>/<sid>/manifest.json` and Parquet for
three tables; PERSONNEL is listed as excluded, not written. `Server(root,
token)` serves `root` with the three behaviours the client depends on:
bearer auth (403 without it), byte ranges (206), and PROPFIND Depth 1
(207). It counts requests and bytes so a test can assert that a query
read a slice, not the file.
"""

import json
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SHOTS = [(1, "2024-06-03 10:00:00", "run1", "ELM study"),
         (2, "2024-06-03 11:00:00", "run1", "elm study, repeat"),
         (3, "2024-07-10 09:30:00", "run2", "calibration")]
SHOTS_TYPE = [(1, "plasma"), (2, "plasma"), (3, "calibration")]
RUNS = [("run1", "ELMs"), ("RUN2", "cal day")]   # RUN2 upper-case on purpose

# SHOTS carries an incompressible NOTES column so its file is several times
# DuckDB's parquet footer prefetch (16 KiB in 1.5). A smaller file is read
# whole by that one prefetch, and "a query reads a slice" could not be told
# from "a query reads the file".
NOTES_BYTES = 32 * 1024


def _notes(shot):
    import hashlib
    out, i = [], 0
    while len(out) * 64 < NOTES_BYTES:
        out.append(hashlib.sha256("{}:{}".format(shot, i).encode()).hexdigest())
        i += 1
    return "".join(out)


def build(root, sid, collation="nocase"):
    import duckdb
    d = os.path.join(root, sid)
    os.makedirs(d, exist_ok=True)
    con = duckdb.connect()
    con.execute("CREATE TABLE SHOTS(SHOT INTEGER, ENTERED TIMESTAMP, RUN VARCHAR, BRIEF VARCHAR, NOTES VARCHAR)")
    con.executemany("INSERT INTO SHOTS VALUES (?, ?, ?, ?, ?)",
                    [row + (_notes(row[0]),) for row in SHOTS])
    con.execute("CREATE TABLE SHOTS_TYPE(shot INTEGER, shot_type VARCHAR)")
    con.executemany("INSERT INTO SHOTS_TYPE VALUES (?, ?)", SHOTS_TYPE)
    con.execute("CREATE TABLE RUNS(RUN VARCHAR, BRIEF VARCHAR)")
    con.executemany("INSERT INTO RUNS VALUES (?, ?)", RUNS)
    tables = []
    for name, rows, pk in (("SHOTS", SHOTS, ["SHOT"]), ("SHOTS_TYPE", SHOTS_TYPE, ["shot"]),
                           ("RUNS", RUNS, ["RUN"])):
        path = os.path.join(d, name + ".parquet")
        con.execute("COPY (SELECT * FROM {} ORDER BY 1) TO '{}' (FORMAT parquet, ROW_GROUP_SIZE 2)".format(name, path))
        tables.append({"name": name, "rows": len(rows), "primary_key": pk, "sorted_by": pk,
                       "read_at": "2026-10-05T12:00:00Z",
                       "files": [{"path": name + ".parquet", "bytes": os.path.getsize(path),
                                  "sha256": "0" * 64, "rows": len(rows)}]})
    manifest = {
        "schema": "fdp-sql-snapshot/1", "id": sid, "created_at": "2026-10-05T12:00:00Z",
        "source": {"server": "fixture", "database": "d3drdb", "shot_ceiling": 3,
                   "isolation": "read committed"},
        "transforms": {"rtrim_char": True, "collation": collation},
        "excluded": [{"table": "PERSONNEL", "reason": "people"}],
        "skipped": [],
        "tables": tables,
    }
    with open(os.path.join(d, "manifest.json"), "w") as fh:
        json.dump(manifest, fh)
    return manifest


class Server:
    """`with Server(root, token) as s: s.url ...`. Thread-safe counters in
    `s.stats` = {"requests", "bytes", "methods"}."""

    def __init__(self, root, token, redirect_to=None):
        self.root = root
        self.token = token
        self.stats = {"requests": 0, "bytes": 0, "methods": []}
        lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _count(self, nbytes=0):
                with lock:
                    outer.stats["requests"] += 1
                    outer.stats["bytes"] += nbytes
                    outer.stats["methods"].append(self.command)

            def _authorized(self):
                auth = self.headers.get("Authorization", "")
                if auth == "Bearer " + outer.token or "authz=" + outer.token in self.path:
                    return True
                self._count()
                self.send_response(403)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return False

            def _local(self):
                rel = self.path.split("?")[0].lstrip("/")
                return os.path.normpath(os.path.join(outer.root, rel))

            def do_HEAD(self):
                if not self._authorized():
                    return
                p = self._local()
                if not os.path.isfile(p):
                    self._count(); self.send_response(404); self.send_header("Content-Length", "0"); self.end_headers(); return
                self._count()
                self.send_response(200)
                self.send_header("Content-Length", str(os.path.getsize(p)))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()

            def do_GET(self):
                if not self._authorized():
                    return
                p = self._local()
                if not os.path.isfile(p):
                    self._count(); self.send_response(404); self.send_header("Content-Length", "0"); self.end_headers(); return
                n = os.path.getsize(p)
                rng = self.headers.get("Range")
                if rng:
                    m = re.match(r"bytes=(\d+)-(\d*)", rng)
                    a = int(m.group(1))
                    e = min(int(m.group(2)) if m.group(2) else n - 1, n - 1)
                else:
                    a, e = 0, n - 1
                length = e - a + 1
                with open(p, "rb") as fh:
                    fh.seek(a)
                    data = fh.read(length)
                # Counted before anything is sent: a client that has its
                # response must see it counted, or a test races the handler.
                self._count(length)
                if rng:
                    self.send_response(206)
                    self.send_header("Content-Range", "bytes {}-{}/{}".format(a, e, n))
                else:
                    self.send_response(200)
                self.send_header("Content-Length", str(length))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                self.wfile.write(data)

            def do_PROPFIND(self):
                if not self._authorized():
                    return
                p = self._local()
                if not os.path.isdir(p):
                    self._count(); self.send_response(404); self.send_header("Content-Length", "0"); self.end_headers(); return
                base = self.path.split("?")[0].rstrip("/")
                hrefs = [base + "/"] + [base + "/" + n for n in sorted(os.listdir(p))]
                body = ('<?xml version="1.0"?><D:multistatus xmlns:D="DAV:">'
                        + "".join("<D:response><D:href>{}</D:href></D:response>".format(h) for h in hrefs)
                        + "</D:multistatus>").encode()
                self._count(len(body))
                self.send_response(207)
                self.send_header("Content-Type", "application/xml")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = "http://127.0.0.1:{}".format(self._srv.server_address[1])

    def __enter__(self):
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self._srv.shutdown()
        self._srv.server_close()

    def reset(self):
        self.stats.update(requests=0, bytes=0, methods=[])
