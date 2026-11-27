"""쌍별 조각 매칭: A 각 조각이 B에서 가장 닮은 조각(사람 조각 제외)과 그 유사도. 원래 모델과 run_g0."""
import sys, json
exec(open('test6.py').read().split("out = dict")[0])
pairs = [tuple(p.split('-')) for p in sys.argv[2:]]
res = {}
with torch.no_grad():
    for tag, sc in (('base', 0.), ('g0', 2.)):
        for m in loras: m.scale = sc
        pf, cls = net.m.backbone(X); P = F.normalize(pf.flatten(2).transpose(1, 2), dim=2)
        for a, b in pairs:
            i, j = names.index(a), names.index(b); S = P[i] @ P[j].T
            S = S.masked_fill(M[j][None], -2)
            v, k = S.max(1); res[f'{tag}|{a}-{b}'] = dict(v=v.cpu().numpy().round(3).tolist(), k=k.cpu().tolist(), ma=M[i].cpu().tolist())
json.dump(res, open('match.json', 'w'))
