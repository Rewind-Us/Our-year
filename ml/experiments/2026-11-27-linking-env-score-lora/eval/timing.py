import sys, time, json, numpy as np, torch
from pathlib import Path
VTD = Path.home() / 'our-year-train/vt'; sys.path.insert(0, str(VTD))
from common import *
from PIL import Image
net = megaloc(4, 8, str(VTD / 'run_g0/step2000.ckpt'))
RD = Path('up'); rn = sorted(json.load(open(RD / 'groups.json')))
X = torch.stack([(torch.from_numpy(np.asarray(Image.open(RD / f'i_{n}.png').convert('RGB'), dtype=np.float32) / 255).permute(2, 0, 1) - MEAN) / STD for n in rn])
def sync(): torch.mps.synchronize() if dev == 'mps' else None
with torch.no_grad():
    net.m.backbone(X[:2].to(dev)); sync()
    t = time.time(); P = torch.cat([F.normalize(net.m.backbone(X[i:i + 8].to(dev))[0].flatten(2).transpose(1, 2), dim=2) for i in range(0, 24, 8)]); sync(); fe = (time.time() - t) / 24
    I, J = zip(*[(i, j) for i in range(24) for j in range(i + 1, 24)]); I, J = list(I), list(J)
    t = time.time()
    for k in range(0, len(I), 64):
        S = P[I[k:k + 64]] @ P[J[k:k + 64]].transpose(1, 2); s = .5 * (S.max(2).values.mean(1) + S.max(1).values.mean(1))
    sync(); pe = (time.time() - t) / len(I)
print(json.dumps(dict(feat_ms_per_img=round(fe * 1000, 1), pair_ms=round(pe * 1000, 3), pairs=len(I))))
