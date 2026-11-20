"""Find which hard_cap statement makes mariadbd grow without bound, in a throwaway container.

The container `wl-capprobe` (started by hand with a 2 GB memory cap) holds one table of
synthetic unit vectors. Each case runs once, on a fresh connection, under max_statement_time,
while a thread samples the container's cgroup memory every 0.1 s. A lost connection with the
container's OOMKilled flag set means the statement ran the server out of its 2 GB.

    python capprobe.py setup N          # (re)create table t with N random vectors
    python capprobe.py run [CASE ...]   # run cases, append results to capprobe_results.jsonl
"""
from __future__ import annotations

import contextlib
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pymysql

HERE = Path(__file__).resolve().parent
import os

CONTAINER = os.environ.get("PROBE_CONTAINER", "wl-capprobe")
PORT = int(os.environ.get("PROBE_PORT", "3307"))
DIM = 384
K = 10
EF = 100
STATEMENT_SECONDS = 60
RESULTS = HERE / os.environ.get("PROBE_RESULTS", "capprobe_results.jsonl")


def password() -> str:
    return (HERE / "capprobe.pw").read_text().strip()


def connect() -> pymysql.Connection:
    return pymysql.connect(host="127.0.0.1", port=PORT, user="root", password=password(),
                           database="probe", autocommit=True, read_timeout=STATEMENT_SECONDS + 30)


def cgroup_file(name: str) -> Path:
    cid = subprocess.check_output(["docker", "inspect", "-f", "{{.Id}}", CONTAINER], text=True).strip()
    return Path(f"/sys/fs/cgroup/system.slice/docker-{cid}.scope/{name}")


def container_state() -> dict:
    out = subprocess.check_output(
        ["docker", "inspect", "-f", "{{.State.Status}} {{.State.OOMKilled}} {{.RestartCount}}", CONTAINER],
        text=True).split()
    return {"status": out[0], "oom_killed": out[1] == "true", "restarts": int(out[2])}


class Monitor(threading.Thread):
    def __init__(self, path: Path) -> None:
        super().__init__(daemon=True)
        self.path, self.peak, self.samples, self.stop = path, 0, [], False

    def run(self) -> None:
        t0 = time.monotonic()
        while not self.stop:
            try:
                v = int(self.path.read_text())
            except (OSError, ValueError):
                v = -1
            self.peak = max(self.peak, v)
            self.samples.append((round(time.monotonic() - t0, 1), v))
            time.sleep(0.1)


def unit(rng: np.random.Generator, n: int) -> np.ndarray:
    x = rng.standard_normal((n, DIM)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def setup(n: int, npy_dir: str | None = None) -> None:
    rng = np.random.default_rng(1)
    if npy_dir:
        parts = sorted(Path(npy_dir).glob("part_*.npy"))
        vecs = np.concatenate([np.load(p, allow_pickle=False) for p in parts]).astype(np.float32)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        n = len(vecs)
    else:
        vecs = unit(rng, n)
    w = rng.integers(0, 100_000, n)
    np.save(HERE / "capprobe_vecs.npy", vecs)
    np.save(HERE / "capprobe_w.npy", w)
    c = connect()
    with c.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS t")
        cur.execute("CREATE TABLE t (id INT UNSIGNED NOT NULL PRIMARY KEY, w INT NOT NULL, "
                    "v VECTOR(384) NOT NULL, VECTOR INDEX (v) M=16 DISTANCE=cosine) ENGINE=InnoDB")
        c.autocommit(False)
        t0 = time.monotonic()
        for i in range(0, n, 1000):
            cur.executemany("INSERT INTO t (id, w, v) VALUES (%s, %s, %s)",
                            [(j + 1, int(w[j]), vecs[j].tobytes()) for j in range(i, min(i + 1000, n))])
        c.commit()
        print(f"inserted {n} rows in {time.monotonic() - t0:.1f} s")
    c.close()


def cases() -> dict[str, tuple[str, tuple]]:
    vecs = np.load(HERE / "capprobe_vecs.npy")
    w = np.load(HERE / "capprobe_w.npy")
    import os
    qi = os.environ.get("Q_INDEX")
    q = np.load(HERE / "capprobe_q.npy")[int(qi)] if qi is not None else unit(np.random.default_rng(2), 1)[0]
    q = (q / np.linalg.norm(q)).astype(np.float32)
    d = np.sort(1.0 - vecs @ q)
    def thr(n: int) -> float:  # exactly n rows have distance < thr
        return float((d[n - 1] + d[n]) / 2)
    ws = np.sort(w)
    def wlim(n: int) -> int:  # rows with w < wlim: at least n
        return int(ws[n - 1] + 1)
    qb = q.tobytes()
    knn = "SELECT id, VEC_DISTANCE_COSINE(v, %s) AS distance FROM t ORDER BY VEC_DISTANCE_COSINE(v, %s) LIMIT %s"
    thr_sql = ("SELECT id, VEC_DISTANCE_COSINE(v, %s) AS distance FROM t WHERE VEC_DISTANCE_COSINE(v, %s) < %s "
               "ORDER BY VEC_DISTANCE_COSINE(v, %s) LIMIT %s")
    sub_sql = ("SELECT x.id, x.distance FROM (" + knn + ") AS x WHERE x.distance < %s ORDER BY x.distance LIMIT %s")
    pred_sql = "SELECT id FROM t WHERE w < %s ORDER BY VEC_DISTANCE_COSINE(v, %s) LIMIT %s"
    return {
        "bare": (knn, (qb, qb, K)),
        "subquery_L100_n3": (sub_sql, (qb, qb, 100, thr(3), K)),
        "threshold_n30": (thr_sql, (qb, qb, thr(30), qb, K)),
        "threshold_n10": (thr_sql, (qb, qb, thr(10), qb, K)),
        "threshold_n3": (thr_sql, (qb, qb, thr(3), qb, K)),
        "predicate_w_100": (pred_sql, (wlim(100), qb, K)),
        "predicate_w_5": (pred_sql, (wlim(5), qb, K)),
    }


def run(names: list[str]) -> None:
    all_cases = cases()
    for name in names or list(all_cases):
        sql, params = all_cases[name]
        state0 = container_state()
        if state0["status"] != "running":
            print(f"{name}: container not running ({state0}); stop")
            return
        mon = Monitor(cgroup_file("memory.current"))
        mem0 = int(cgroup_file("memory.current").read_text())
        rec = {"case": name, "n_rows_table": None, "mem_before": mem0}
        c = connect()
        with c.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM t")
            rec["n_rows_table"] = cur.fetchone()[0]
            cur.execute("SET SESSION mhnsw_ef_search = %s", (EF,))
            cur.execute("SET SESSION max_statement_time = %s", (STATEMENT_SECONDS,))
            cur.execute("EXPLAIN " + sql, params)
            rec["explain"] = [list(map(str, r)) for r in cur.fetchall()]
            mon.start()
            t0 = time.monotonic()
            try:
                cur.execute(sql, params)
                rows = cur.fetchall()
                rec.update(outcome="ok", rows=len(rows))
            except pymysql.err.MySQLError as e:
                rec.update(outcome=f"error {e.args[0]}", message=str(e.args[1])[:200])
            rec["seconds"] = round(time.monotonic() - t0, 2)
        mon.stop = True
        mon.join()
        with contextlib.suppress(pymysql.err.Error):
            c.close()
        time.sleep(1)
        rec["peak_mem"] = mon.peak
        rec["mem_samples_every_1s"] = mon.samples[::10]
        rec["state_after"] = container_state()
        RESULTS.open("a").write(json.dumps(rec) + "\n")
        print(f"{name}: {rec.get('outcome')} rows={rec.get('rows')} {rec['seconds']} s "
              f"peak {mon.peak / 2**20:.0f} MiB (before {mem0 / 2**20:.0f}) state {rec['state_after']}")
        if rec["state_after"]["status"] != "running":
            print("container stopped; not running further cases")
            return


if __name__ == "__main__":
    if sys.argv[1] == "setup":
        setup(int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else None)
    elif sys.argv[1] == "run":
        run(sys.argv[2:])


def repeat(name: str, times: int) -> None:
    """Run one case `times` times on ONE connection; memory after each run."""
    sql, params = cases()[name]
    path = cgroup_file("memory.current")
    mon = Monitor(path)
    mon.start()
    c = connect()
    out = {"repeat": name, "times": times, "mem_before": int(path.read_text()), "runs": []}
    with c.cursor() as cur:
        cur.execute("SET SESSION mhnsw_ef_search = %s", (EF,))
        cur.execute("SET SESSION max_statement_time = %s", (STATEMENT_SECONDS,))
        for i in range(times):
            t0 = time.monotonic()
            try:
                cur.execute(sql, params)
                r = {"rows": len(cur.fetchall())}
            except pymysql.err.MySQLError as e:
                r = {"error": f"{e.args[0]} {str(e.args[1])[:120]}"}
            time.sleep(1)
            try:
                mem_now = int(path.read_text())
            except OSError:
                mem_now = -1  # the container is gone
            r.update(seconds=round(time.monotonic() - t0 - 1, 2), mem_after=mem_now, peak=mon.peak)
            out["runs"].append(r)
            print(f"{name} run {i + 1}: {r} -> {r['mem_after'] / 2**20:.0f} MiB, peak {mon.peak / 2**20:.0f} MiB", flush=True)
            RESULTS.open("a").write(json.dumps({"repeat_run": name, "i": i + 1, **r}) + "\n")
            if "error" in r or mem_now < 0:
                break
    with contextlib.suppress(pymysql.err.Error):
        c.close()
    time.sleep(2)
    try:
        out["mem_after_close"] = int(path.read_text())
    except OSError:
        out["mem_after_close"] = None
    out["state_after"] = container_state()
    mon.stop = True
    RESULTS.open("a").write(json.dumps(out) + "\n")
    print("after close:", out["mem_after_close"] and out["mem_after_close"] / 2**20, out["state_after"])


if __name__ == "__main__" and sys.argv[1] == "repeat":
    repeat(sys.argv[2], int(sys.argv[3]))
