"""ETH3D로 MegaLoc의 시점 변화 한계와 사람 마스킹 효과를 재요.
1. 같은 장면 사진 쌍: 시선 각도 차이, 카메라 거리, 공유 3D 점 비율(겹침)별 유사도
2. 다른 장면 쌍: '같은 시간대, 다른 장소'의 대용
3. 사람 합성: 같은 사람 + 다른 장소(오탐 위험), 다른 사람 + 같은 장소(찾아야 함)를 원본, 사람 있음, 사람 마스킹 세 조건으로 비교
"""
import json, random, itertools, numpy as np, torch
from pathlib import Path
from PIL import Image
from common import *
R = ROOT / 'pubeval/raw/eth3d/x'; OUT = VT / 'eth3d'; OUT.mkdir(exist_ok=True)
SAME_ROOM = [{'relief', 'relief_2'}]
def place(s):
    for g in SAME_ROOM:
        if s in g: return sorted(g)[0]
    return s

def qrot(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def load_scene(sd):
    cal = next(sd.glob('dslr_calibration_jpg/images.txt'))
    L = [l for l in open(cal) if not l.startswith('#')]
    ims = []
    for a, b in zip(L[0::2], L[1::2]):
        f = a.split(); q = np.array(list(map(float, f[1:5]))); t = np.array(list(map(float, f[5:8]))); Rm = qrot(q)
        pts = set(int(v) for v in b.split()[2::3] if int(v) >= 0)
        ims.append(dict(path=str(sd / 'images' / f[9]), center=-Rm.T @ t, dir=Rm.T @ np.array([0, 0, 1.]), pts=pts))
    return ims

scenes = sorted(p for p in R.iterdir() if (p / 'dslr_calibration_jpg').exists())
items = []
for sd in scenes:
    for it in load_scene(sd): it['scene'] = sd.name; it['place'] = place(sd.name); items.append(it)
print('scenes', [s.name for s in scenes], 'images', len(items), flush=True)

people = [p for p in sorted((VT / 'people').glob('*.png')) if (lambda im: im.size[1] > 1.2 * im.size[0])(Image.open(p))]
print('people', len(people), flush=True)

def compose(path, person, cover, seed):
    """사람을 화면 높이의 cover 비율 크기로 아래쪽에 붙여요. 정사각 322 기준 마스크도 함께 반환."""
    rng = random.Random(seed)
    bg = Image.open(path).convert('RGB'); bg.draft('RGB', (1000, 1000)); bg = bg.resize((644, 644), Image.BICUBIC)
    p = Image.open(person); h = int(644 * cover); w = int(p.size[0] * h / p.size[1]); p = p.resize((w, h), Image.BICUBIC)
    x = rng.randint(0, max(0, 644 - w)); y = 644 - h
    bg.paste(p, (x, y), p)
    m = Image.new('L', (644, 644)); m.paste(p.split()[3], (x, y))
    return bg, m

def tens(im, mask=None):
    im = im.resize((322, 322), Image.BICUBIC)
    x = (torch.from_numpy(np.asarray(im, dtype=np.float32).copy() / 255).permute(2, 0, 1) - MEAN) / STD
    if mask is not None:
        mm = torch.from_numpy(np.asarray(mask.resize((322, 322), Image.NEAREST)) > 127); x[:, mm] = 0
    return x

@torch.no_grad()
def emb(net, xs, bs=16):
    out = [net(torch.stack(xs[i:i + bs]).to(dev)).float().cpu() for i in range(0, len(xs), bs)]
    return F.normalize(torch.cat(out), dim=1).numpy()

net = megaloc()
def clean(path):
    im = Image.open(path).convert('RGB'); im.draft('RGB', (1000, 1000)); return im
E = emb(net, [tens(clean(it['path'])) for it in items])

# 1. 같은 장면 쌍과 다른 장소 쌍(사람 없음)
pos, neg = [], []
for i, j in itertools.combinations(range(len(items)), 2):
    a, b = items[i], items[j]; s = float(E[i] @ E[j])
    if a['place'] == b['place']:
        if a['scene'] != b['scene']: continue
        ang = float(np.degrees(np.arccos(np.clip(a['dir'] @ b['dir'], -1, 1))))
        ov = len(a['pts'] & b['pts']) / max(1, min(len(a['pts']), len(b['pts'])))
        pos.append(dict(i=i, j=j, s=s, ang=ang, dist=float(np.linalg.norm(a['center'] - b['center'])), ov=ov))
    else: neg.append(s)
neg = np.array(neg)
thr = {k: float(np.quantile(neg, q)) for k, q in (('fpr1', .99), ('fpr01', .999))}
def table(rows, key, bins):
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        r = [x for x in rows if lo <= x[key] < hi]
        if not r: continue
        s = np.array([x['s'] for x in r])
        out.append(dict(bin=f'{lo}-{hi}', n=len(r), mean_sim=round(float(s.mean()), 3), p10=round(float(np.quantile(s, .1)), 3),
                        pass_fpr1=round(float((s > thr['fpr1']).mean()), 3), pass_fpr01=round(float((s > thr['fpr01']).mean()), 3)))
    return out
rep = dict(n_pos=len(pos), n_neg=len(neg), neg_mean=round(float(neg.mean()), 3), neg_p99=round(thr['fpr1'], 3), neg_p999=round(thr['fpr01'], 3), neg_max=round(float(neg.max()), 3),
           by_angle=table(pos, 'ang', [0, 15, 30, 45, 60, 90, 120, 180.1]),
           by_overlap=table(pos, 'ov', [0, .01, .05, .1, .2, .4, .7, 1.01]),
           by_dist_m=table(pos, 'dist', [0, 1, 2, 4, 8, 16, 1000]))
print(json.dumps(rep), flush=True)

# 3. 사람 합성: 겹침이 있는 같은 장면 쌍 300개, 다른 장소 쌍 300개
rng = random.Random(0)
P = [x for x in pos if x['ov'] >= .05]; P = rng.sample(P, min(300, len(P)))
idx = list(range(len(items))); N = []
while len(N) < 300:
    i, j = rng.sample(idx, 2)
    if items[i]['place'] != items[j]['place']: N.append((i, j))
res = {}
for cover in (.5, .8):
    for cond in ('person', 'masked'):
        sp, sn = [], []
        for k, x in enumerate(P):  # 다른 사람, 같은 장소
            pa, pb = rng.sample(people, 2)
            A = compose(items[x['i']]['path'], pa, cover, k); B = compose(items[x['j']]['path'], pb, cover, k + 9999)
            sp.append((A, B))
        for k, (i, j) in enumerate(N):  # 같은 사람, 다른 장소
            pa = rng.choice(people)
            sn.append((compose(items[i]['path'], pa, cover, k), compose(items[j]['path'], pa, cover, k + 7777)))
        def sims(pairs):
            xs = []
            for A, B in pairs:
                xs += [tens(A[0], A[1] if cond == 'masked' else None), tens(B[0], B[1] if cond == 'masked' else None)]
            e = emb(net, xs); return (e[0::2] * e[1::2]).sum(1)
        a, b = sims(sp), sims(sn)
        res[f'cover{cover}_{cond}'] = dict(same_place_diff_person_mean=round(float(a.mean()), 3), diff_place_same_person_mean=round(float(b.mean()), 3),
                                           diff_place_same_person_p99=round(float(np.quantile(b, .99)), 3),
                                           recall_at_clean_fpr1=round(float((a > thr['fpr1']).mean()), 3), fp_rate_at_clean_fpr1=round(float((b > thr['fpr1']).mean()), 3),
                                           auc=round(float((a[:, None] > b[None]).mean()), 4))
        print(cover, cond, json.dumps(res[f'cover{cover}_{cond}']), flush=True)
# 사람 없는 같은 쌍 기준
a = np.array([x['s'] for x in P]); b = np.array([float(E[i] @ E[j]) for i, j in N])
res['clean'] = dict(same_place_mean=round(float(a.mean()), 3), diff_place_mean=round(float(b.mean()), 3), recall_at_fpr1=round(float((a > thr['fpr1']).mean()), 3),
                    fp_rate_at_fpr1=round(float((b > thr['fpr1']).mean()), 3), auc=round(float((a[:, None] > b[None]).mean()), 4))
print('clean', json.dumps(res['clean']), flush=True)
rep['person_test'] = res
json.dump(rep, open(OUT / 'report.json', 'w'), indent=1)
json.dump(pos, open(OUT / 'pos_pairs.json', 'w'))
print('DONE', flush=True)
