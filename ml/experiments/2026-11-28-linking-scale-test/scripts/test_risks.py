import sys, json, collections
import numpy as np
exec(open('test_pipeline.py').read().split("for mode in ('random'")[0])
def run(tt, label, c=cfg):
    photos = [PL.Photo(str(i), owner[i], None if tt[i] is None else float(tt[i])) for i in range(N)]
    res = PL.link(photos, V, lambda p: [S[i, j] for i, j in p], c)
    places = collections.defaultdict(list)
    for i in range(N): places[g[i]].append(i)
    real_both = {k: v for k, v in places.items() if not k.startswith('pano:') and {owner[i] for i in v} == {'a', 'b'}}
    whole = [k for k, v in real_both.items() if any({str(i) for i in v} <= set(m['photos']) for m in res.moments)]
    linked = [k for k, v in real_both.items() if any(len({owner[int(p)] for p in m['photos'] if g[int(p)] == k}) == 2 for m in res.moments)]
    mixed = sum(len({g[int(p)] for p in m['photos']}) > 1 for m in res.moments)
    print(f'{label:42s} 두 사람이 이어진 실사진 장소 {len(linked)}/{len(real_both)}, 통째로 {len(whole)}, 섞인 순간 {mixed}, 시각 없음 {len(res.no_time)}')
tt = [float(x) for x in t]
for x in sorted({x for x in g if x.startswith('pano:')}):
    b = mins('2026-06-15 00:00:00') + rng2.uniform(0, 170 * 1440)
    for i in range(N):
        if g[i] == x: tt[i] = b + rng2.uniform(0, 15)
run(tt, '기본')
for sk in (10, 20, 45, 90, 600):
    t2 = [v + sk if owner[i] == 'b' else v for i, v in enumerate(tt)]; run(t2, f'친구 시계가 {sk}분 늦음')
t2 = [v + 9 * 60 if owner[i] == 'b' else v for i, v in enumerate(tt)]; run(t2, '친구 사진만 시간대 9시간 어긋남(UTC 저장)')
t2 = [v + 3 * 1440 if owner[i] == 'b' and g[i] == 'cafe' else v for i, v in enumerate(tt)]; run(t2, '카페: 친구가 3일 뒤 같은 곳에서 찍음')
t2 = [None if owner[i] == 'b' and g[i] in ('rest', 'cafe') else v for i, v in enumerate(tt)]; run(t2, '식당, 카페 친구 사진 시각 없음(카톡 전달)')
