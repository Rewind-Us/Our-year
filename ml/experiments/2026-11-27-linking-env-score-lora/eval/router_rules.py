"""규칙 기반 방식 선택. 기준값은 미리 정한 굵은 값만 쓴다(실사진으로 세밀 조정하지 않음).
1) 쌍의 사람 비율 최대가 40% 이상 -> 3단계(영역 비교)로 넘김, 여기서는 보류
2) 직접 점수 > T -> 같은 장소
3) 아니면 같은 시간대 다른 사진을 다리로: 두 다리 점수 모두 > T + 0.01 이면 같은 장소"""
import json, numpy as np, sys
from PIL import Image
d = json.loads(open('res24.txt').read()); n = d['names']; grp = json.load(open('real/up/groups.json'))
pf = np.array([(np.asarray(Image.open(f'real/up/m_{k}.png').convert('L')) > 127).mean() for k in n])
S = np.array(d['g0_env']); N = len(n); T, TB, PH = .629, .639, .40
res = {}
for i in range(N):
    for j in range(i + 1, N):
        if max(pf[i], pf[j]) >= PH: r = 'hold'
        elif S[i, j] > T: r = 'direct'
        elif any(S[i, k] > TB and S[k, j] > TB for k in range(N) if k not in (i, j)): r = 'bridge'
        else: r = 'no'
        res[(i, j)] = r
same = lambda i, j: grp[n[i]] == grp[n[j]]
for lab, cond in (('같은 장소', True), ('다른 장소', False)):
    ks = [k for k in res if same(*k) == cond]
    print(lab, len(ks), {r: sum(res[k] == r for k in ks) for r in ('direct', 'bridge', 'no', 'hold')})
print('보류 쌍:', [(n[i], n[j], grp[n[i]] if same(i, j) else 'neg') for (i, j), r in res.items() if r == 'hold' and same(i, j)])
json.dump({f'{n[i]}|{n[j]}': r for (i, j), r in res.items()}, open('router.json', 'w'))
