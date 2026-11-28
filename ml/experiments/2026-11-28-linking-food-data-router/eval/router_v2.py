"""라우터 v2: 직접 -> 다리 -> 확인 질문 후보.

direct : 두 사진의 같은 환경 점수 > T 이면 자동 연결
bridge : 같은 후보 묶음의 다른 사진 k에 대해 S[i,k] > TB 이고 S[k,j] > TB 이면 자동 연결 (TB = T + OFF)
ask    : 후보 묶음 안에서 두 사진이 서로 1등이고, 각자 2등과의 차이 >= MG, 점수 >= FLOOR 이면
         "같은 곳에서 찍었나요?" 확인 질문 후보. 자동 연결하지 않는다(2026-11-28 결정).
no     : 연결하지 않음

사용: python router_v2.py scores.json groups.json T [OFF=0.02] [MG=0.03] [FLOOR=0]
scores.json 은 eval/score_real_pairs.py 출력({"names": [...], "g0_env": NxN}).
groups.json 은 {사진이름: 장소 라벨}.
"""
import json, sys
import numpy as np


def route(S, T, TB, MG, FLOOR, idx):
    idx = list(idx)
    res = {}
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            if S[i, j] > T:
                res[(i, j)] = 'direct'
                continue
            if any(S[i, k] > TB and S[k, j] > TB for k in idx if k not in (i, j)):
                res[(i, j)] = 'bridge'
                continue
            oi = sorted((S[i, k] for k in idx if k != i), reverse=True)
            oj = sorted((S[j, k] for k in idx if k != j), reverse=True)
            mutual = S[i, j] >= oi[0] and S[i, j] >= oj[0]
            gap = min(oi[0] - (oi[1] if len(oi) > 1 else 0), oj[0] - (oj[1] if len(oj) > 1 else 0))
            res[(i, j)] = 'ask' if mutual and gap >= MG and S[i, j] >= FLOOR else 'no'
    return res


if __name__ == '__main__':
    env, grpf = sys.argv[1], sys.argv[2]
    T = float(sys.argv[3])
    OFF, MG, FLOOR = (list(map(float, sys.argv[4:7])) + [0.02, 0.03, 0.0][len(sys.argv[4:7]):])
    d = json.load(open(env)); n = d['names']; S = np.array(d['g0_env']); grp = json.load(open(grpf))
    r = route(S, T, T + OFF, MG, FLOOR, range(len(n)))
    for lab, c in (('same', True), ('diff', False)):
        ks = [k for k in r if (grp[n[k[0]]] == grp[n[k[1]]]) == c]
        print(lab, len(ks), {x: sum(r[k] == x for k in ks) for x in ('direct', 'bridge', 'ask')})
