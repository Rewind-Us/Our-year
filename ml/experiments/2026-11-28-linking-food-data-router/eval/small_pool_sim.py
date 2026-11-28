"""후보가 적을 때 상대 순위 규칙이 안전한지 보는 시뮬레이션.
실제 앱은 촬영 시각으로 후보를 먼저 좁히므로 한 번에 비교하는 사진이 몇 장뿐이다.
전체 사진에서 k장을 무작위로 뽑아 router_v2를 돌리는 것을 2,000번 반복하고,
ask 판정이 실제 같은 장소였는지(TP) 아닌지(FP)를 센다.

사용: python small_pool_sim.py scores.json groups.json T out.json
"""
import json, sys
import numpy as np
from router_v2 import route

env, grpf, T, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
d = json.load(open(env)); n = d['names']; S = np.array(d['g0_env']); N = len(n); grp = json.load(open(grpf))
rng = np.random.default_rng(0)
res = []
for k in (3, 5, 8):
    for floor, mg in ((0, .03), (.5, .03), (.55, .03), (.58, .05)):
        tp = fp = fpd = nd = 0
        for _ in range(2000):
            idx = rng.choice(N, k, replace=False)
            for (i, j), v in route(S, T, T + .02, mg, floor, idx).items():
                s = grp[n[i]] == grp[n[j]]
                nd += not s
                if v == 'ask':
                    tp += s; fp += not s
                elif v != 'no':
                    fpd += not s
        r = dict(pool=k, floor=floor, gap=mg, ask_tp=int(tp), ask_fp=int(fp), auto_fp_rate=round(fpd / nd, 4))
        res.append(r); print(r)
json.dump(res, open(out, 'w'), indent=1)
