"""같은 장소 파노라마 묶기: 이름 토큰이 겹치면 같은 그룹(보수적, 과하게 묶여도 안전). 출력 pano/groups.json {id: group}."""
import json, re, itertools, collections
from pathlib import Path
P = Path.home() / 'our-year-train/pubeval/raw/pano'
m = json.load(open(P / 'meta.json')); ids = [x['id'] for x in m]
STOP = set(('wm ph jpg 360 360view panorama panoramic full spherical image interior innenansicht innenaufnahme verkleinert view inside the of and '
            'de la le du des der die das im in st saint church kirche dorfkirche photosphere img pano panos x5b 360x180g 360pano via poly haven '
            'photos degree room hall house museum old new upper lower 01 02 03 cc').split())
def tok(i):
    s = i[3:].lower()
    return {t for t in re.split(r'[_\W]+', s) if t and t not in STOP and not t.isdigit() and len(t) > 2}
T = {i: tok(i) for i in ids}; df = collections.Counter(t for s in T.values() for t in s)
par = {i: i for i in ids}
def f(x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
for a, b in itertools.combinations(ids, 2):
    A, B = T[a], T[b]; c = A & B
    if not c: continue
    ov = len(c) / min(len(A), len(B))
    if ov >= .5 and (len(c) >= 2 or min(df[t] for t in c) <= 8): par[f(a)] = f(b)
g = {i: f(i) for i in ids}
json.dump(g, open(P / 'groups.json', 'w'), ensure_ascii=False, indent=0)
sz = collections.Counter(g.values())
print('ids', len(ids), 'groups', len(sz), 'multi', sum(v > 1 for v in sz.values()), 'largest', sz.most_common(5))
for r, n in sz.most_common():
    if n > 1: print(n, sorted(i for i in ids if g[i] == r)[:6])
