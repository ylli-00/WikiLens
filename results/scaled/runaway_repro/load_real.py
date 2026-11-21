"""Load the first N real text embeddings into table t (and the query vector into qv) for the repro.

The vectors: wikilense_s1_v1.tar.gz from https://github.com/ylli-00/WikiLens/releases/tag/data-s1-v1,
folder wikilense_s1/vectors/, files part_*.npy (float32, 384 dimensions, BAAI/bge-small-en-v1.5).
The query vector: query_vector.json (384 floats), next to this script.

    python3 load_real.py --vectors DIR --query query_vector.json --rows 80000 \
        --host 127.0.0.1 --port 3306 --user root --password ... --database test
Then restart the server and run, from outside its container:
    mariadb -h 127.0.0.1 -P 3306 -u root -p test < 2_query.sql
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pymysql

ap = argparse.ArgumentParser()
ap.add_argument("--vectors", required=True)
ap.add_argument("--query", required=True)
ap.add_argument("--rows", type=int, default=80000)
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=3306)
ap.add_argument("--user", default="root")
ap.add_argument("--password", default="")
ap.add_argument("--database", default="test")
a = ap.parse_args()

parts, have = [], 0
for p in sorted(Path(a.vectors).glob("part_*.npy")):
    parts.append(np.load(p, allow_pickle=False).astype(np.float32))
    have += len(parts[-1])
    if have >= a.rows:
        break
v = np.concatenate(parts)[: a.rows]
n = len(v)
q = np.array(json.loads(Path(a.query).read_text()), dtype=np.float32)

c = pymysql.connect(host=a.host, port=a.port, user=a.user, password=a.password, database=a.database,
                    autocommit=True)
with c.cursor() as cur:
    cur.execute("DROP TABLE IF EXISTS t, qv")
    cur.execute("CREATE TABLE t (id INT UNSIGNED NOT NULL PRIMARY KEY, w INT NOT NULL, v VECTOR(384) NOT NULL, "
                "VECTOR INDEX (v) M=16 DISTANCE=cosine) ENGINE=InnoDB")
    for i in range(0, n, 1000):  # w is a permutation of 0..n-1, so exactly k rows have w < k
        cur.executemany("INSERT INTO t (id, w, v) VALUES (%s, %s, %s)",
                        [(j + 1, (j + 1) * 7919 % n, v[j].tobytes()) for j in range(i, min(i + 1000, n))])
    cur.execute("CREATE TABLE qv (v VECTOR(384) NOT NULL)")
    cur.execute("INSERT INTO qv VALUES (%s)", (q.tobytes(),))
    cur.execute("SELECT COUNT(*) FROM t")
    print("rows in t:", cur.fetchone()[0])
c.close()
