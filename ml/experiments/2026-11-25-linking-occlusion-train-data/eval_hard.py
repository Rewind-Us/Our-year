#!/usr/bin/env python3
"""ITLP 교차 시간대 검색에서 MegaLoc을 비교해요. C 원본, S와 L 인물 합성, R 같은 면적 사각형, E1 마스킹, E3 LoRA.

사용법: python3 eval_hard.py --itlp <itlp> --cutouts <폴더> --model <megaloc> [--lora last.ckpt] --out <폴더>
"""
import argparse, csv, json, math, random, subprocess, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
from safetensors.torch import load_file
from build_occlusion_train import place
from train_lora import LoRA, MEAN, STD

BANDS = {'S': (.10, .25), 'L': (.40, .60)}

def tensor(im, mask=None):
    x = (torch.from_numpy(np.asarray(im.convert('RGB').resize((322, 322), Image.BICUBIC), dtype=np.float32).copy() / 255).permute(2, 0, 1) - MEAN) / STD
    if mask is not None: x[:, torch.from_numpy(np.asarray(mask.resize((322, 322), Image.NEAREST)).copy() > 0)] = 0
    return x

def occlude(bg, cuts, band, rng):
    """인물 1~2명을 넣어 가림 면적을 맞춰요. 실패하면 None이에요."""
    W, H = bg.size; lo, hi = BANDS[band]
    for _ in range(60):
        k = 1 if band == 'S' else rng.choice([1, 2])
        group = rng.sample(cuts, k); t = rng.uniform(lo + .02, hi - .02)
        axes = [rng.uniform(.3, .7)] if k == 1 else [rng.uniform(.24, .36), rng.uniform(.64, .76)]
        layer, ok = Image.new('RGBA', (W, H)), True
        for c, ax in zip(group, axes):
            _, w, h, x, y = place(c, W, H, t if k == 1 else t * .55, ax, rng.uniform(.02, .1))
            if (H - y) / h < .45: ok = False; break
            one = Image.new('RGBA', (W, H)); one.paste(c.resize((w, h), Image.LANCZOS), (x, y)); layer = Image.alpha_composite(layer, one)
        m = layer.getchannel('A').point(lambda v: 255 if v > 0 else 0); n = m.histogram()[255] / (W * H)
        if ok and lo <= n < hi:
            person = Image.composite(layer.convert('RGB'), bg, layer.getchannel('A'))
            x0, y0, x1, y1 = m.getbbox(); asp = (x1 - x0) / (y1 - y0); rh = min(H, math.sqrt(n * W * H / asp)); rw = min(W, n * W * H / rh)
            left = int(min(max(0, (x0 + x1) / 2 - rw / 2), W - rw)); rect = bg.copy()
            rect.paste((124, 116, 104), (left, int(H - rh), int(left + rw), H))
            return person, m, rect, n
    return None

def main():
    p = argparse.ArgumentParser()
    for k in ('itlp', 'cutouts', 'model', 'out', 'lora'): p.add_argument('--' + k, type=Path)
    a = p.parse_args(); a.out.mkdir(exist_ok=True); dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    if dev == 'mps': torch.mps.set_per_process_memory_fraction(.35)
    sys.path.insert(0, str(a.model)); from megaloc_model import MegaLoc
    net = MegaLoc(); net.load_state_dict(load_file(a.model / 'model.safetensors')); net.eval().requires_grad_(False).to(dev)
    G = [r['path'] for r in csv.DictReader(open(a.itlp / 'gallery.csv'))]; Q = list(csv.DictReader(open(a.itlp / 'queries.csv')))
    gi = {g: i for i, g in enumerate(G)}; cuts = [Image.open(f).convert('RGBA') for f in sorted(a.cutouts.glob('*.png'))]
    op = lambda q: Image.open(a.itlp / 'img' / q).convert('RGB')
    @torch.no_grad()
    def embed(gen):
        out, buf = [], []
        for t in gen:
            buf.append(t)
            if len(buf) == 8: out.append(net(torch.stack(buf).to(dev)).cpu()); buf = []
        if buf: out.append(net(torch.stack(buf).to(dev)).cpu())
        return torch.cat(out)
    T = {k: [] for k in ('C', 'S', 'L', 'RS', 'RL', 'E1_S', 'E1_L')}; cov = []; fail = 0
    for i, q in enumerate(Q):
        bg = op(q['path']); bg.thumbnail((640, 640)); T['C'].append(tensor(bg)); c = {}
        for b in BANDS:
            o = occlude(bg, cuts, b, random.Random(f'{q["path"]}{b}'))
            if o is None: raise SystemExit(f'occlusion failed {b} {q["path"]}')
            per, m, rect, n = o; c[b] = n
            T[b].append(tensor(per)); T['R' + b].append(tensor(rect)); T['E1_' + b].append(tensor(per, m))
            if i < 8: per.save(a.out / f'ex_{i}_{b}.jpg')
        cov.append(c)
    def score(E, GE):
        s = E @ GE.T; top = s.topk(5, 1).indices.tolist(); r1 = []; r5 = []; fl = []
        for q, t in zip(Q, top):
            pos = {gi[x] for x in q['positives'].split('|')}
            r1.append(t[0] in pos); r5.append(bool(pos & set(t))); fl.append(G[t[0]].split('/')[1] == f"floor_{q['floor']}")
        return dict(r1=sum(r1), r5=sum(r5), same_floor_top1=sum(fl)), r1
    rep = dict(gallery=len(G), queries=len(Q), coverage={b: round(float(np.mean([c[b] for c in cov])), 3) for b in BANDS})
    hits = {}
    def run(tag):
        GE = embed(tensor(op(g)) for g in G)
        for k in T:
            rep[f'{tag}/{k}'], hits[f'{tag}/{k}'] = score(embed(iter(T[k])), GE); print(tag, k, rep[f'{tag}/{k}'], flush=True)
    run('E0')
    if a.lora:
        for b in net.backbone.blocks[-4:]: b.attn.qkv = LoRA(b.attn.qkv, 8, 16).to(dev).requires_grad_(False)
        r = net.load_state_dict(torch.load(a.lora, map_location=dev), strict=False); assert not r.unexpected_keys
        run('E3')
    base = hits['E0/C']
    rep['lost_vs_E0C'] = {k: sum(b and not h for b, h in zip(base, v)) for k, v in hits.items()}
    rep['gained_vs_E0C'] = {k: sum(h and not b for b, h in zip(base, v)) for k, v in hits.items()}
    (a.out / 'report.json').write_text(json.dumps(rep, indent=1)); print(json.dumps({k: rep[k] for k in ('coverage', 'lost_vs_E0C', 'gained_vs_E0C')}))

if __name__ == '__main__': main()
