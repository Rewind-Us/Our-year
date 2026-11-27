"""실사진 테스트: 원래 MegaLoc vs LoRA run_g0(step2000)의 같은 환경 점수(사람 조각 제외 chamfer)."""
import sys, json, glob, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path.home() / 'our-year-train/vt'))
from common import *
from PIL import Image
D = Path(sys.argv[1]); G = 23
names = sorted(p.stem[2:] for p in D.glob('i_*.png'))
def tens(p): return (torch.from_numpy(np.asarray(Image.open(p).convert('RGB'), dtype=np.float32) / 255).permute(2, 0, 1) - MEAN) / STD
def pmask(p):
    m = (np.asarray(Image.open(p).convert('L')) > 127).astype(np.float32)
    return torch.from_numpy(m[:G * 14, :G * 14].reshape(G, 14, G, 14).mean((1, 3)) > .5).flatten()
X = torch.stack([tens(D / f'i_{n}.png') for n in names]).to(dev); M = torch.stack([pmask(D / f'm_{n}.png') for n in names]).to(dev)
def chamfer_mat(A, B, ma, mb):
    S = torch.einsum('ipd,jqd->ijpq', A, B)
    S = S.masked_fill(mb[None, :, None, :], -2).masked_fill(ma[:, None, :, None], -2)
    va, vb = (~ma).float(), (~mb).float()
    ab = (S.max(3).values * va[:, None]).sum(2) / va.sum(1)[:, None].clamp(min=1)
    ba = (S.max(2).values.clamp(min=-1) * vb[None]).sum(2) / vb.sum(1)[None].clamp(min=1)
    return .5 * (ab + ba)
net = megaloc(4, 8, str(Path.home() / 'our-year-train/vt/run_g0/step2000.ckpt'))
loras = [m for m in net.modules() if isinstance(m, LoRA)]
out = dict(names=names)
with torch.no_grad():
    for tag, sc in (('base', 0.), ('g0', 2.)):
        for m in loras: m.scale = sc
        pf, cls = net.m.backbone(X); g = F.normalize(net.m.aggregator((pf, cls)), dim=1)
        P = F.normalize(pf.flatten(2).transpose(1, 2), dim=2)
        out[tag + '_env'] = chamfer_mat(P, P, M, M).cpu().numpy().round(3).tolist()
        out[tag + '_glob'] = (g @ g.T).cpu().numpy().round(3).tolist()
print(json.dumps(out))
