#!/usr/bin/env python3
"""ITLP Campus Indoor 좌표로 교차 시간대 검색 평가 목록을 만들어요. 이미지는 내려받지 않아요.

검색 대상은 night, 질의는 twilight의 앞 카메라예요. 층이 같고 2m, 30도 안이면 정답이에요.
질의마다 정답 중 가장 가까운 사진도 0.5m 이상 또는 10도 이상 떨어지게 골라 같은 시점을 피해요.
"""
import csv, json, math, random, sys
from pathlib import Path

def yaw(r):
    x, y, z, w = (float(r[k]) for k in ('qx', 'qy', 'qz', 'qw'))
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
def gap(a, b):
    d = math.hypot(float(a['tx']) - float(b['tx']), float(a['ty']) - float(b['ty']))
    return d, abs((yaw(a) - yaw(b) + 180) % 360 - 180)
def main(src, out, per_floor=20, seed=0):
    R = [r for r in csv.DictReader(open(src))]
    for r in R: r['path'] = f"{r['track']}/floor_{r['floor']}/front_cam/{r['front_cam_ts']}.png"
    G = [r for r in R if 'night' in r['track']]
    rng, Q = random.Random(seed), []
    stats = {}
    for f in sorted({r['floor'] for r in R}):
        gal, cand = [g for g in G if g['floor'] == f], []
        for q in (r for r in R if 'twilight' in r['track'] and r['floor'] == f):
            pos = [(*gap(q, g), g['path']) for g in gal]
            near = [p for p in pos if p[0] <= 2 and p[1] <= 30]
            if near and all(p[0] >= .5 or p[1] >= 10 for p in near):
                cand.append((q, near))
        # 바로 이웃한 질의가 겹치지 않게 1.5m 이상 떨어뜨려요.
        rng.shuffle(cand); picked = []
        for q, near in cand:
            if len(picked) < per_floor and all(gap(q, p)[0] >= 1.5 or gap(q, p)[1] > 60 for p, _ in picked): picked.append((q, near))
        stats[f] = dict(gallery=len(gal), eligible=len(cand), picked=len(picked))
        Q += picked
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    with open(out / 'gallery.csv', 'w', newline='') as h:
        w = csv.writer(h); w.writerow(['path', 'floor']); w.writerows((g['path'], g['floor']) for g in G)
    with open(out / 'queries.csv', 'w', newline='') as h:
        w = csv.writer(h); w.writerow(['path', 'floor', 'positives', 'nearest_m'])
        for q, near in Q: w.writerow([q['path'], q['floor'], '|'.join(p[2] for p in near), round(min(p[0] for p in near), 2)])
    print(json.dumps(stats), 'queries', len(Q), 'mean positives', sum(len(n) for _, n in Q) / max(1, len(Q)))
if __name__ == '__main__': main(*sys.argv[1:3])
