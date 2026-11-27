"""3단계 후보: 영역 비교. 사진마다 사람 아닌 조각 특징을 k-means(k=12)로 묶어 영역을 만들고,
두 사진 영역끼리 서로 가장 닮은 짝(mutual NN)의 유사도 상위 3개 평균을 점수로 쓴다.
겹치는 물체가 일부만 있어도 그 부분만 보도록 하는 것이 목적. 비교용으로 조각 상위 20% 평균(top20)도 계산.
기준값: ETH3D 다른 장소 오탐 1%. 평가: ETH3D 각도별 + 실사진 24장."""
import sys, json, itertools, numpy as np, torch
from pathlib import Path
VTD = Path.home() / 'our-year-train/vt'; sys.path.insert(0, str(VTD))
from common import *
from PIL import Image
G = 23; K = 12
net = megaloc(4, 8, str(VTD / 'run_g0/step2000.ckpt'))
def tens(a): return (torch.from_numpy(a.astype(np.float32) / 255).permute(2, 0, 1) - MEAN) / STD
@torch.no_grad()
def feats(X): return torch.cat([F.normalize(net.m.backbone(X[i:i + 16].to(dev))[0].flatten(2).transpose(1, 2), dim=2).half().cpu() for i in range(0, len(X), 16)])
RD = Path(sys.argv[1]); grp = json.load(open(RD / 'groups.json')); rn = sorted(grp)
RX = torch.stack([tens(np.asarray(Image.open(RD / f'i_{n}.png').convert('RGB'))) for n in rn])
def pm(n):
    m = (np.asarray(Image.open(RD / f'm_{n}.png').convert('L')) > 127).astype(np.float32)
    return torch.from_numpy(m[:G * 14, :G * 14].reshape(G, 14, G, 14).mean((1, 3)) > .5).flatten()
RM = torch.stack([pm(n) for n in rn])
@torch.no_grad()
def regions(P, M):
    out = []
    g = torch.Generator().manual_seed(0)
    for p, m in zip(P, M):
        x = p[~m].float().to(dev); k = min(K, len(x))
        c = x[torch.randperm(len(x), generator=g)[:k].to(dev)]
        for _ in range(15):
            a = (x @ c.T).argmax(1)
            c = torch.stack([F.normalize(x[a == t].mean(0), dim=0) if (a == t).any() else c[t] for t in range(k)])
        out.append(c.half().cpu())
    return out
def reg_score(Ra, Rb):
    S = Ra.float() @ Rb.float().T
    mut = (S == S.max(1, keepdim=True).values) & (S == S.max(0, keepdim=True).values)
    v = S[mut].sort(descending=True).values
    return float(v[:3].mean()) if len(v) else 0.
@torch.no_grad()
def top20(P, M, pairs):
    out = []
    for k in range(0, len(pairs), 64):
        I, J = map(list, zip(*pairs[k:k + 64])); A, B = P[I].float().to(dev), P[J].float().to(dev); ma, mb = M[I].to(dev), M[J].to(dev)
        S = (A @ B.transpose(1, 2)).masked_fill(mb[:, None, :], -2).masked_fill(ma[:, :, None], -2)
        sa = S.max(2).values.masked_fill(ma, -9).sort(1, descending=True).values; sb = S.max(1).values.masked_fill(mb, -9).sort(1, descending=True).values
        na, nb = (~ma).sum(1), (~mb).sum(1)
        r = [.5 * (sa[t, :max(1, int(na[t] * .2))].mean() + sb[t, :max(1, int(nb[t] * .2))].mean()) for t in range(len(I))]
        out.append(torch.stack(r).cpu())
    return torch.cat(out).numpy()
PR = feats(RX); RR = regions(PR, RM)
rpairs = [(i, j) for i, j in itertools.combinations(range(len(rn)), 2)]
out = {'names': rn, 'region': {f'{rn[i]}|{rn[j]}': round(reg_score(RR[i], RR[j]), 4) for i, j in rpairs},
       'top20': {f'{rn[i]}|{rn[j]}': round(float(v), 4) for (i, j), v in zip(rpairs, top20(PR, RM, rpairs))}}
json.dump(out, open(sys.argv[2], 'w')); print('DONE')
