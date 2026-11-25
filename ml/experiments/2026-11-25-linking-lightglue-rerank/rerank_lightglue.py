#!/usr/bin/env python3
"""MegaLoc 상위 5개를 ALIKED와 LightGlue 일치점 수로 다시 정렬해요. 규칙은 결과를 보기 전에 정했어요.

규칙 A: 일치점이 많은 순서, 같으면 MegaLoc 순서. 규칙 B: 최다 일치점이 20개 미만이면 MegaLoc 순서를 유지.
rot0은 회전 없음, rot4는 질의를 0, 90, 180, 270도 돌린 값 중 최댓값이에요. 로봇 팔 영역의 특징점은 빼요.
"""
import argparse, csv, json, random, sys, time
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image
from lightglue import ALIKED, LightGlue
from lightglue.utils import numpy_image_to_torch, rbd
from safetensors.torch import load_file
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / '2026-11-25-linking-occlusion-train-data'))
from eval_hard import occlude, tensor

p = argparse.ArgumentParser()
for k in ('itlp', 'cutouts', 'model', 'out'): p.add_argument('--' + k, type=Path, required=True)
p.add_argument('--limit', type=int, default=0); a = p.parse_args(); a.out.mkdir(exist_ok=True)
dev = 'mps'; torch.mps.set_per_process_memory_fraction(.4)
sys.path.insert(0, str(a.model)); from megaloc_model import MegaLoc
G = [r['path'] for r in csv.DictReader(open(a.itlp / 'gallery.csv'))]; Q = list(csv.DictReader(open(a.itlp / 'queries.csv')))
if a.limit: Q = Q[:a.limit]
def op(x):
    im = Image.open(a.itlp / 'img' / x).convert('RGB'); im.thumbnail((640, 640)); return im
cuts = [Image.open(f).convert('RGBA') for f in sorted(a.cutouts.glob('*.png'))]
conds = {'C': [], 'S': [], 'L': []}
for q in Q:
    bg = op(q['path']); conds['C'].append((bg, None))
    for b in 'SL':
        per, m, _, _ = occlude(bg, cuts, b, random.Random(f'{q["path"]}{b}')); conds[b].append((per, m))
cache = a.out / 'gallery.pt'
net = MegaLoc(); net.load_state_dict(load_file(a.model / 'model.safetensors')); net.eval().requires_grad_(False).to(dev)
@torch.no_grad()
def embed(it):
    o, b = [], []
    for x in it:
        b.append(x)
        if len(b) == 8: o.append(net(torch.stack(b).to(dev)).cpu()); b = []
    if b: o.append(net(torch.stack(b).to(dev)).cpu())
    return torch.cat(o)
GE = torch.load(cache) if cache.exists() else embed(tensor(Image.open(a.itlp / 'img' / g)) for g in G)
torch.save(GE, cache)
tops = {c: (embed(tensor(i) for i, _ in v) @ GE.T).topk(5, 1).indices.tolist() for c, v in conds.items()}
del net; torch.mps.empty_cache()
ex, lg = ALIKED(max_num_keypoints=1024).eval().to(dev), LightGlue(features='aliked').eval().to(dev)
def feat(im, mask=None, rot=0):
    W, H = im.size; keep = np.ones((H, W), bool); keep[int(.70 * H):, int(.33 * W):int(.60 * W)] = False
    if mask is not None: keep &= ~(np.asarray(mask) > 0)
    ar, kp = np.asarray(im), keep
    for _ in range(rot // 90): ar, kp = np.rot90(ar), np.rot90(kp)
    with torch.no_grad(): f = ex.extract(numpy_image_to_torch(ar.copy()).to(dev), resize=None)
    k = f['keypoints'][0].cpu().numpy().astype(int); ok = torch.from_numpy(kp[k[:, 1].clip(0, kp.shape[0] - 1), k[:, 0].clip(0, kp.shape[1] - 1)].copy()).to(dev)
    for n in ('keypoints', 'descriptors', 'keypoint_scores'): f[n] = f[n][:, ok]
    return {n: f[n].cpu() for n in ('keypoints', 'descriptors', 'image_size')}
def inliers(f, g):
    if min(f['keypoints'].shape[1], g['keypoints'].shape[1]) < 16: return 0
    with torch.no_grad(): m = lg({'image0': {k: v.to(dev) for k, v in f.items()}, 'image1': {k: v.to(dev) for k, v in g.items()}})['matches'][0].cpu()
    if len(m) < 8: return 0
    cv2.setRNGSeed(0)
    try: _, k = cv2.findFundamentalMat(f['keypoints'][0][m[:, 0]].numpy(), g['keypoints'][0][m[:, 1]].numpy(), cv2.USAC_MAGSAC, 2., .999, 5000)
    except cv2.error: return 0  # 점 배치가 나빠 모델을 못 구한 경우예요.
    return 0 if k is None else int(k.sum())
gc, rep, rows, t0 = {}, {}, [], time.time()
for c, mode in (('C', 'full'), ('S', 'full'), ('S', 'person_removed'), ('L', 'full'), ('L', 'person_removed')):
    hit = {k: 0 for k in ('megaloc', 'A_rot0', 'A_rot4', 'B_rot0', 'B_rot4', 'top5')}; pi, ni = [], []
    for qi, (q, (im, m)) in enumerate(zip(Q, conds[c])):
        pos = set(q['positives'].split('|')); top = [G[i] for i in tops[c][qi]]
        fq = [feat(im, m if mode != 'full' else None, r) for r in (0, 90, 180, 270)]; sc = []
        for g in top:
            if g not in gc: gc[g] = feat(op(g))
            sc.append([inliers(f, gc[g]) for f in fq])
        torch.mps.empty_cache()
        ok = [g in pos for g in top]; hit['megaloc'] += ok[0]; hit['top5'] += any(ok)
        for tag, v in (('rot0', [s[0] for s in sc]), ('rot4', [max(s) for s in sc])):
            o = max(range(5), key=lambda i: (v[i], -i)); hit['A_' + tag] += ok[o]; hit['B_' + tag] += ok[o if v[o] >= 20 else 0]
        for s, k in zip(sc, ok): (pi if k else ni).append(s[0])
        rows.append(dict(cond=c, mode=mode, query=q['path'], ok=ok, inliers=sc))
    rep[f'{c}/{mode}'] = dict(hit, n=len(Q), pos_inliers_median=float(np.median(pi)) if pi else None, neg_inliers_median=float(np.median(ni)) if ni else None)
    print('RR', c, mode, rep[f'{c}/{mode}'], f'{time.time() - t0:.0f}s', flush=True)
(a.out / 'report.json').write_text(json.dumps(dict(summary=rep, rows=rows)))
print('RRDONE', flush=True)
