"""대량 처리 검증용 특징 계산: 실사진 60장 + 학습에 안 쓴 시험 파노라마에서 자른 사진.
출력 stress.npz: names, groups, S(같은 환경 점수), Gl(전역 벡터 코사인), person ratio."""
import sys, json, hashlib, random
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
H = Path.home() / 'our-year-train'; sys.path.insert(0, str(H / 'vt'))
from common import megaloc, MEAN, STD, dev
import pano_crop
PANO = pano_crop.PANO; FOOD = H / 'pubeval/raw/pano_food'
ids = sorted(x.stem for x in PANO.glob('*.jpg'))
GRP = json.load(open(PANO.parent / 'groups.json'))
ids = [i for i in ids if not (i.startswith('wm_') and 'Poly_Haven' in i)]
FM = json.load(open(FOOD / 'food_meta.json'))
GRP = {**{i: GRP.get(i, i) for i in ids}, **FM['groups']}; ids = ids + FM['keep']
te = [i for i in ids if int(hashlib.md5(GRP[i].encode()).hexdigest(), 16) % 10 == 0]
print('test panos', len(te), flush=True)
rng = random.Random(0); G = 23
names, groups, X, M = [], [], [], []
def tens(a): return (torch.from_numpy(a.astype(np.float32) / 255).permute(2, 0, 1) - MEAN) / STD
for pid in te:
    eq = np.asarray(Image.open((FOOD / 'img' if pid.startswith('wf_') else PANO) / f'{pid}.jpg').convert('RGB'))
    if eq.shape[1] > 4096:
        eq = np.asarray(Image.fromarray(eq).resize((4096, 2048), Image.BILINEAR))
    for k in range(6):
        im = pano_crop.crop(eq, rng.uniform(0, 360), rng.uniform(-10, 10), 70.)
        names.append(f'p_{pid[:40]}_{k}'); groups.append('pano:' + GRP[pid]); X.append(tens(im)); M.append(torch.zeros(G * G, dtype=torch.bool))
real = Path.home() / 'rtmp8/all60'; rg = json.load(open(real / 'groups.json'))
for n in sorted(rg):
    a = np.asarray(Image.open(real / f'i_{n}.png').convert('RGB'))
    m = (np.asarray(Image.open(real / f'm_{n}.png').convert('L')) > 127).astype(np.float32)
    names.append(n); groups.append(rg[n]); X.append(tens(a)); M.append(torch.from_numpy(m.reshape(G, 14, G, 14).mean((1, 3)) > .5).flatten())
print('photos', len(names), flush=True)
net = megaloc(4, 8, str(H / 'vt/run_f1/step2000.ckpt'))
Ps, Gs = [], []
with torch.no_grad():
    for i in range(0, len(X), 16):
        xb = torch.stack(X[i:i + 16]).to(dev); pf, cls = net.m.backbone(xb)
        Gs.append(F.normalize(net.m.aggregator((pf, cls)), dim=1).cpu())
        Ps.append(F.normalize(pf.flatten(2).transpose(1, 2), dim=2).half().cpu())
P = torch.cat(Ps).to(dev); Gl = torch.cat(Gs); Mk = torch.stack(M).to(dev)
def chamfer_mat(A, B, ma, mb):
    S = torch.einsum('ipd,jqd->ijpq', A, B)
    S = S.masked_fill(mb[None, :, None, :], -2).masked_fill(ma[:, None, :, None], -2)
    va, vb = (~ma).to(A.dtype), (~mb).to(A.dtype)
    ab = (S.max(3).values * va[:, None]).sum(2) / va.sum(1)[:, None].clamp(min=1)
    ba = (S.max(2).values.clamp(min=-1) * vb[None]).sum(2) / vb.sum(1)[None].clamp(min=1)
    return .5 * (ab + ba)
import time; t = time.time()
with torch.no_grad():
    S = torch.cat([chamfer_mat(P[i:i + 4], P, Mk[i:i + 4], Mk).float().cpu() for i in range(0, len(P), 4)])
torch.mps.synchronize(); print('pairs', len(P) ** 2, 'sec', round(time.time() - t, 1), flush=True)
np.savez_compressed(H / 'app/stress.npz', names=np.array(names), groups=np.array(groups), S=S.numpy(), Gl=(Gl @ Gl.T).numpy())
print('saved')
