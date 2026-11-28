"""잇기 v1 판정기: 같은 환경 점수 행렬을 받아 사진 쌍마다 direct, bridge, ask, no 를 돌려준다.

설정은 같은 폴더의 config.json 을 읽는다. 기준값을 바꿀 때는 이 코드가 아니라 config.json 만 고친다.

사용: python router.py scores.json [config.json]
scores.json 형식: {"names": [사진 이름...], "g0_env": N x N 점수}
"""
import json, sys
from pathlib import Path
import numpy as np


def load_config(path=None):
    return json.load(open(path or Path(__file__).with_name('config.json')))


def route(S, cfg, idx=None):
    r = cfg['router']; T, TB = r['direct_threshold'], r['bridge_threshold']
    MG, FLOOR = r['ask']['min_gap'], r['ask']['min_score']
    idx = list(range(len(S))) if idx is None else list(idx)
    res = {}
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            if S[i, j] > T:
                res[(i, j)] = 'direct'; continue
            if any(S[i, k] > TB and S[k, j] > TB for k in idx if k not in (i, j)):
                res[(i, j)] = 'bridge'; continue
            oi = sorted((S[i, k] for k in idx if k != i), reverse=True)
            oj = sorted((S[j, k] for k in idx if k != j), reverse=True)
            mutual = S[i, j] >= oi[0] and S[i, j] >= oj[0]
            gap = min(oi[0] - (oi[1] if len(oi) > 1 else 0), oj[0] - (oj[1] if len(oj) > 1 else 0))
            res[(i, j)] = 'ask' if mutual and gap >= MG and S[i, j] >= FLOOR else 'no'
    return res


def groups(res, n):
    """자동 연결(direct, bridge)로 이어진 사진 묶음. 앱에서 '같은 공간' 하나로 보여 줄 단위."""
    par = list(range(n))
    def f(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for (i, j), v in res.items():
        if v in ('direct', 'bridge'): par[f(i)] = f(j)
    out = {}
    for i in range(n): out.setdefault(f(i), []).append(i)
    return list(out.values())


if __name__ == '__main__':
    d = json.load(open(sys.argv[1])); cfg = load_config(sys.argv[2] if len(sys.argv) > 2 else None)
    n = d['names']; S = np.array(d['g0_env']); res = route(S, cfg)
    print('같은 공간 묶음:', [[n[i] for i in g] for g in groups(res, len(n)) if len(g) > 1])
    print('확인 질문 후보:', [(n[i], n[j]) for (i, j), v in res.items() if v == 'ask'])
