"""pipeline.py 검증: 348장(실사진 60 + 파노라마 조각 288)을 두 사람으로 나눠 돌린다."""
import sys, json, time, collections
import numpy as np
sys.path.insert(0, sys.argv[1]); import pipeline as PL
exec(open('analyze.py').read().split("full=np.ones")[0])   # n, g, S, Gl, N
w, U = np.linalg.eigh(Gl); w = np.clip(w, 0, None); V = U * np.sqrt(w)   # Gl = V V^T
cfg = PL.load_config()
import datetime as dt
src = open('analyze.py').read(); exec(src[src.index('known={'):src.index('def run_time')])   # base, t, mins
rng2 = np.random.default_rng(1)
owner = {}
cnt = collections.Counter()
for i in range(N):
    owner[i] = 'ab'[cnt[g[i]] % 2]; cnt[g[i]] += 1
def scenario(mode, gap=None):
    tt = t.copy(); rb = list({base[x] for x in real_groups})
    for x in sorted({x for x in g if x.startswith('pano:')}):
        b = (mins('2026-06-15 00:00:00') + rng2.uniform(0, 170 * 1440)) if mode == 'random' else rb[rng2.integers(len(rb))] + rng2.uniform(-90, 90)
        for i in range(N):
            if g[i] == x: tt[i] = b + rng2.uniform(0, 15)
    photos = [PL.Photo(str(i), owner[i], float(tt[i])) for i in range(N)]
    c = json.loads(json.dumps(cfg))
    if gap: c['pipeline']['event_gap_min'] = gap
    calls = []
    def score_fn(pairs):
        calls.append(len(pairs)); return [S[i, j] for i, j in pairs]
    t0 = time.time(); res = PL.link(photos, V, score_fn, c); el = time.time() - t0
    mixed = [m for m in res.moments if len({g[int(p)] for p in m['photos']}) > 1]
    # 두 사람 사진이 모두 있는 실제 장소 중, 한 순간으로 통째로 묶인 수
    places = collections.defaultdict(list)
    for i in range(N): places[g[i]].append(i)
    both = {k: v for k, v in places.items() if {owner[i] for i in v} == {'a', 'b'}}
    whole = {k for k, v in both.items() if any({str(i) for i in v} <= set(m['photos']) for m in res.moments)}
    real_both = [k for k in both if not k.startswith('pano:')]
    ask_ok = sum(g[int(a)] == g[int(b)] for a, b, _ in res.ask)
    print(f"{mode:8s} 간격 {c['pipeline']['event_gap_min']:3d}분 | 비교 묶음 {res.stats['pools']:2d}, 정밀 비교 {res.stats['pairs_scored']:5d}쌍 (나-친구 전체 {res.stats['pairs_all_cross']}) | "
          f"실사진 장소 통째로 {len([k for k in real_both if k in whole])}/{len(real_both)}, 파노라마 장소 {len([k for k in both if k.startswith('pano:') and k in whole])}/{len(both)-len(real_both)} | "
          f"섞인 순간 {len(mixed)} | 확인 질문 {len(res.ask)} (정답 {ask_ok}) | {el:.2f}s")
    return res
for mode in ('random', 'overlap'):
    for gap in (60, 120, 240):
        scenario(mode, gap)
