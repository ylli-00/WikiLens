"""Does max_statement_time stop the runaway vector search?

Runs one statement RUNS times on one connection, under max_statement_time = LIMIT_S, against a
throwaway MariaDB container `wl-hardcap` on port 3308 that holds S1 (never the project's server),
and appends the outcomes, times and the container's memory peaks to tl_check_results.jsonl.
The query vector is the first row of capprobe_q.npy next to this file.

    python tl_check.py [LIMIT_S] [CASE] [RUNS]
"""
import contextlib
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pymysql

from wikilense import db
from wikilense.config import load_settings

LIMIT_S = int(sys.argv[1]) if len(sys.argv) > 1 else 10
CASE = sys.argv[2] if len(sys.argv) > 2 else "threshold3"
RUNS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
HERE = Path(__file__).resolve().parent
CONTAINER = "wl-hardcap"

settings = load_settings()
assert settings.db_port == 3308, "the throwaway server only"
cid = subprocess.check_output(["docker", "inspect", "-f", "{{.Id}}", CONTAINER], text=True).strip()
MEM = Path(f"/sys/fs/cgroup/system.slice/docker-{cid}.scope/memory.current")
peak = [0]
stop = [False]


def monitor() -> None:
    while not stop[0]:
        with contextlib.suppress(OSError, ValueError):
            peak[0] = max(peak[0], int(MEM.read_text()))
        time.sleep(0.1)


def connect() -> pymysql.Connection:
    for _ in range(90):
        try:
            return db.connect(settings)
        except pymysql.err.MySQLError:
            time.sleep(2)
    raise SystemExit("server did not come back")


threading.Thread(target=monitor, daemon=True).start()
qb = db.vec_param(np.load(HERE / "capprobe_q.npy")[0])
conn = connect()
with conn.cursor() as cur:
    cur.execute("SELECT VEC_DISTANCE_COSINE(embedding, %s) AS d FROM chunk IGNORE INDEX (embedding) "
                "ORDER BY d LIMIT 4", (qb,))
    d = [r[0] for r in cur.fetchall()]
thr = (d[2] + d[3]) / 2  # exactly 3 chunks are nearer than this
SQL = {
    "threshold3": (
        ("SELECT chunk_id FROM chunk WHERE VEC_DISTANCE_COSINE(embedding, %s) < %s "
         "ORDER BY VEC_DISTANCE_COSINE(embedding, %s) LIMIT 10"),
        (qb, thr, qb),
    ),
}
sql, params = SQL[CASE]
out = []
for i in range(RUNS):
    peak[0] = 0
    t0 = time.monotonic()
    try:
        with conn.cursor() as cur:
            cur.execute("SET SESSION mhnsw_ef_search = 100")
            cur.execute("SET SESSION max_statement_time = %s", (LIMIT_S,))
            cur.execute(sql, params)
            r = {"rows": len(cur.fetchall())}
    except pymysql.err.MySQLError as e:
        r = {"error": f"{type(e).__name__} {getattr(e, 'args', [''])[0]} {str(e)[:100]}"}
    r.update(run=i + 1, seconds=round(time.monotonic() - t0, 2), peak_mib=round(peak[0] / 2**20))
    time.sleep(2)
    try:
        r["mem_after_mib"] = round(int(MEM.read_text()) / 2**20)
    except OSError:
        r["mem_after_mib"] = None
    r["state"] = subprocess.check_output(
        ["docker", "inspect", "-f",
         "{{.State.Status}} oom={{.State.OOMKilled}} restarts={{.RestartCount}}", CONTAINER],
        text=True).strip()
    print(json.dumps(r), flush=True)
    out.append(r)
    if "error" in r and "1969" not in r["error"]:
        with contextlib.suppress(pymysql.err.Error):
            conn.close()
        conn = connect()
stop[0] = True
with open(HERE / "tl_check_results.jsonl", "a") as fh:
    fh.write(json.dumps({"limit_s": LIMIT_S, "case": CASE, "thr": thr, "runs": out}) + "\n")
