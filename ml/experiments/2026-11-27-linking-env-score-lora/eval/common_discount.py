"""흔한 패턴 깎기: 조각마다 '흔함 정도'(외부 파노라마 조각 은행에서 가장 닮은 정도)를 재고,
같은 환경 점수의 조각별 유사도에서 lam * 흔함을 뺀 뒤 평균. lam=0이 기존 점수.
평가: ETH3D(다른 장소 오탐 1% 기준선, 각도별) + 실사진 24장(장소 5곳)."""
import sys, json, random, hashlib, itertools, numpy as np, torch
from pathlib import Path
VTD = Path.home() / 'our-year-train/vt'; sys.path.insert(0, str(VTD))
from common import *
from pano_crop import PANO, pano, crop
from PIL import Image
G = 23; LAMS = (0., .25, .5, .75, 1.)
net = megaloc(4, 8, str(VTD / 'run_g0/step2000.ckpt')); loras = [m for m in net.modules() if isinstance(m, LoRA)]
def tens(a): return (torch.from_numpy(a.astype(np.float32) / 255).permute(2, 0, 1) - MEAN) / STD
@torch.no_grad()
def feats(X):
    return torch.cat([F.normalize(net.m.backbone(X[i:i + 16].to(dev))[0].flatten(2).transpose(1, 2), dim=2).half().cpu() for i in range(0, len(X), 16)])
# 은행: 평가용(md5%10==0) 파노라마에서 사람 없이 자른 조각 200장. 학습에 안 쓴 파노라마.
SKY = json.load(open(PANO.parent / 'sky.json'))
ids = sorted(x.stem for x in PANO.glob('*.jpg') if SKY.get(x.stem, 0) <= .2)
te = [i for i in ids if int(hashlib.md5(i.encode()).hexdigest(), 16) % 10 == 0]
rng = np.random.default_rng(5); BX = []
for pid in te:
    eq = pano(pid)
    for _ in range(max(1, 200 // len(te) + 1)): BX.append(tens(crop(eq, rng.uniform(0, 360), rng.uniform(-10, 10), rng.uniform(60, 80))))
BX = torch.stack(BX[:200]); print('bank panos', len(te), 'crops', len(BX), flush=True)
# ETH3D
exec(open(VTD / 'eth3d_exp.py').read().split('people = ')[0].split('scenes = ')[0])
scenes = sorted(q for q in R.iterdir() if (q / 'dslr_calibration_jpg').exists()); items = []
for sd in scenes:
    for it in load_scene(sd): it['scene'] = sd.name; it['place'] = place(sd.name); items.append(it)
EX = torch.stack([img(it['path']) for it in items]); epos = json.load(open(VTD / 'eth3d/pos_pairs.json'))
nr = np.random.default_rng(0); eneg = []
while len(eneg) < 20000:
    i, j = nr.integers(0, len(items), 2)
    if items[i]['place'] != items[j]['place']: eneg.append((int(i), int(j)))
# 실사진
RD = Path(sys.argv[1]); grp = json.load(open(RD / 'groups.json')); rn = sorted(grp)
RX = torch.stack([tens(np.asarray(Image.open(RD / f'i_{n}.png').convert('RGB'))) for n in rn])
def pm(n):
    m = (np.asarray(Image.open(RD / f'm_{n}.png').convert('L')) > 127).astype(np.float32)
    return torch.from_numpy(m[:G * 14, :G * 14].reshape(G, 14, G, 14).mean((1, 3)) > .5).flatten()
RM = torch.stack([pm(n) for n in rn])
rpos = [(i, j) for i, j in itertools.combinations(range(len(rn)), 2) if grp[rn[i]] == grp[rn[j]]]
rneg = [(i, j) for i, j in itertools.combinations(range(len(rn)), 2) if grp[rn[i]] != grp[rn[j]]]
@torch.no_grad()
def commonness(P, Bk):
    out = []
    for i in range(len(P)): out.append((P[i].float().to(dev) @ Bk.T).max(1).values.cpu())
    return torch.stack(out)
@torch.no_grad()
def scores(P, Cm, M, pairs, lam):
    out = []
    for k in range(0, len(pairs), 64):
        I, J = map(list, zip(*pairs[k:k + 64])); A, B = P[I].float().to(dev), P[J].float().to(dev)
        ma, mb = M[I].to(dev), M[J].to(dev); ca, cb = Cm[I].to(dev), Cm[J].to(dev)
        S = (A @ B.transpose(1, 2)).masked_fill(mb[:, None, :], -2).masked_fill(ma[:, :, None], -2)
        sa = S.max(2).values - lam * ca; sb = S.max(1).values - lam * cb
        va, vb = (~ma).float(), (~mb).float()
        out.append((.5 * ((sa * va).sum(1) / va.sum(1).clamp(min=1) + (sb * vb).sum(1) / vb.sum(1).clamp(min=1))).cpu())
    return torch.cat(out).numpy()
def auc(p, n): p, n = np.array(p), np.array(n); return float((p[:, None] > n[None]).mean() + .5 * (p[:, None] == n[None]).mean())
res = {}
for tag, sc in (('base', 0.), ('g0', 2.)):
    for m in loras: m.scale = sc
    Bk = feats(BX).flatten(0, 1).float().to(dev); PE, PR = feats(EX), feats(RX)
    CE, CR = commonness(PE, Bk), commonness(PR, Bk); del Bk
    EM = torch.zeros(len(PE), G * G, dtype=torch.bool); ang = np.array([x['ang'] for x in epos]); ep = [(x['i'], x['j']) for x in epos]
    for lam in LAMS:
        sn = scores(PE, CE, EM, eneg, lam); t = float(np.quantile(sn, .99)); sp = scores(PE, CE, EM, ep, lam)
        e = {f'{lo}-{hi}': round(float((sp[(ang >= lo) & (ang < hi)] > t).mean()), 3) for lo, hi in ((0, 30), (30, 60), (60, 90), (90, 120), (120, 181))}
        rp, rq = scores(PR, CR, RM, rpos, lam), scores(PR, CR, RM, rneg, lam)
        r = dict(auc=round(auc(rp, rq), 3), pass_eth_thr=f'{int((rp > t).sum())}/{len(rp)}', fp_eth_thr=f'{int((rq > t).sum())}/{len(rq)}', above_max_neg=f'{int((rp > rq.max()).sum())}/{len(rp)}')
        bygrp = {}
        for (i, j), v in zip(rpos, rp): bygrp.setdefault(grp[rn[i]], []).append(v > t)
        r['by_place'] = {g: f'{sum(v)}/{len(v)}' for g, v in bygrp.items()}
        res[f'{tag} lam={lam}'] = dict(eth3d=e, eth_auc=round(auc(sp, sn[:3000]), 3), real=r); print(tag, lam, json.dumps(res[f'{tag} lam={lam}'], ensure_ascii=False), flush=True)
json.dump(res, open('discount.json', 'w'), indent=1); print('DONE')
