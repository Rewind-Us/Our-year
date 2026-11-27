"""시점 튜닝 실험 공용: MegaLoc(+LoRA) 로드, 이미지 로드, EgoFoodPlaces 방문 내 시점 변화 평가."""
import json, sys, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from PIL import Image
ROOT = Path.home() / 'our-year-train'; VT = ROOT / 'vt'; EGO = ROOT / 'pubeval/raw/ego/x'
sys.path.insert(0, str(ROOT / 'pubeval')); from models import Mega, MEAN, STD
dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
GAP = 180  # 같은 방문에서 3분 이상 떨어진 사진만 정답으로 봐요(거의 같은 장면 제외)

class LoRA(nn.Module):
    def __init__(s, base, rank, alpha):
        super().__init__(); s.base, s.scale = base, alpha / rank
        s.a, s.b = nn.Linear(base.in_features, rank, bias=False), nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(s.a.weight, a=5 ** .5); nn.init.zeros_(s.b.weight)
    def forward(s, x): return s.base(x) + s.scale * s.b(s.a(x))

def megaloc(lora_blocks=0, rank=8, ckpt=None):
    net = Mega().eval().requires_grad_(False)
    if lora_blocks:
        for blk in net.m.backbone.blocks[-lora_blocks:]: blk.attn.qkv = LoRA(blk.attn.qkv, rank, 2 * rank)
    if ckpt: print('load', net.load_state_dict(torch.load(ckpt, map_location='cpu'), strict=False).unexpected_keys)
    return net.to(dev)

def img(path, size=322, aug=False):
    im = Image.open(path).convert('RGB')
    if aug:
        w, h = im.size; s = random.uniform(.7, 1); cw, ch = int(w * s), int(h * s)
        x0, y0 = random.randint(0, w - cw), random.randint(0, h - ch); im = im.crop((x0, y0, x0 + cw, y0 + ch))
        if random.random() < .5: im = im.transpose(Image.FLIP_LEFT_RIGHT)
    im = im.resize((size, size), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(im, dtype=np.float32).copy() / 255).permute(2, 0, 1)
    if aug: x = (x * random.uniform(.8, 1.2) + random.uniform(-.1, .1)).clamp(0, 1)
    return (x - MEAN) / STD

@torch.no_grad()
def embed(net, paths, bs=16):
    out = []
    for i in range(0, len(paths), bs):
        out.append(net(torch.stack([img(p) for p in paths[i:i + bs]]).to(dev)).float().cpu())
    if dev == 'mps': torch.mps.empty_cache()
    return torch.cat(out)

def ego_test_set(max_per_visit=40, seed=0):
    vs = [v for v in json.load(open(VT / 'ego_visits.json')) if v['split'] == 'test']
    rng = random.Random(seed); rows = []
    for vi, v in enumerate(vs):
        ims = v['imgs'] if len(v['imgs']) <= max_per_visit else sorted(rng.sample(v['imgs'], max_per_visit), key=lambda x: x[1])
        rows += [dict(path=str(EGO / p), t=t, visit=vi, cls=v['cls']) for p, t in ims]
    return rows

def eval_ego(E, rows):
    """질의마다 같은 방문의 3분 이내 사진은 갤러리에서 빼고, 3분 이상 떨어진 같은 방문 사진이 1위인지 봐요."""
    E = F.normalize(E, dim=1); S = (E @ E.T).numpy()
    vis = np.array([r['visit'] for r in rows]); t = np.array([r['t'] for r in rows]); cls = np.array([r['cls'] for r in rows])
    r1, r5, wrong_same_cls, n = 0, 0, 0, 0
    for q in range(len(rows)):
        same = vis == vis[q]; near = same & (np.abs(t - t[q]) < GAP)
        pos = same & ~near
        if not pos.any(): continue
        s = S[q].copy(); s[near] = -9
        top = np.argsort(-s)[:5]; n += 1
        r1 += pos[top[0]]; r5 += pos[top].any()
        wrong_same_cls += (not pos[top[0]]) and cls[top[0]] == cls[q]
    return dict(n=n, r1=round(r1 / n, 4), r5=round(r5 / n, 4), wrong_top1_same_class=round(wrong_same_cls / n, 4))
