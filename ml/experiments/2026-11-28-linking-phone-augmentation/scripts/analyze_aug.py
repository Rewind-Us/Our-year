"""a1~a3와 v1을 실사진 60장으로 비교하고 원인을 나눠 본다.
입력(저장소 밖): up/groups.json(사진별 장소), up/i_*.png, up/m_*.png(사람 마스크), env_{f1,a1,a2,a3}.json(같은 환경 점수),
cases_private.json({"heldout_groups": [...], "hard_pairs": {"이름": [사진A, 사진B]}}).
출력: real_eval.json 과 같은 숫자, 그리고 "어두우면 a2로 바꾸기" 실험 결과.
"""
import json, sys, numpy as np
from PIL import Image
sys.path.insert(0, '../../../linking/v1')   # ml/linking/v1/router.py
import router as _R


def route(M, T, TB, MG, FLOOR, idx):
    cfg = {'router': {'direct_threshold': T, 'bridge_threshold': TB, 'ask': {'min_gap': MG, 'min_score': FLOOR}}}
    return _R.route(M, cfg, idx)

g = json.load(open('up/groups.json')); C = json.load(open('cases_private.json')); HELD = set(C['heldout_groups'])
MODELS = {'v1': ('f1', .611), 'a1': ('a1', .600), 'a2': ('a2', .605), 'a3': ('a3', .617)}
S = {k: np.array(json.load(open(f'env_{f}.json'))['g0_env']) for k, (f, _) in MODELS.items()}
n = json.load(open('env_f1.json'))['names']; N = len(n)
iu = np.triu_indices(N, 1); same = np.array([[g[a] == g[b] for b in n] for a in n])[iu]


def brightness(x):
    im = np.asarray(Image.open(f'up/i_{x}.png').convert('L')).astype(float)
    m = np.asarray(Image.open(f'up/m_{x}.png').convert('L')) > 127
    return im[~m].mean() if (~m).sum() else im.mean()


br = np.array([brightness(x) for x in n])


def sets(M, T):
    out = {}
    for name, sel in (('seen38', lambda x: g[x] not in HELD), ('heldout22', lambda x: g[x] in HELD)):
        idx = [k for k in range(N) if sel(n[k])]; r = route(M, T, T + .02, .03, 0, idx)
        out[name] = dict(found=sum(v in ('direct', 'bridge') for (a, b), v in r.items() if g[n[a]] == g[n[b]]),
                         fp=sum(v in ('direct', 'bridge') for (a, b), v in r.items() if g[n[a]] != g[n[b]]))
    return out


dark = np.maximum(br[iu[0]], br[iu[1]]) < np.quantile(br, .35)
for k, (f, T) in MODELS.items():
    s = S[k][iu]
    print(k, 'AUC', round(float((s[same][:, None] > s[~same][None, :]).mean()), 4), 'own', sets(S[k], T), '0.611', sets(S[k], .611),
          'dark diff mean', round(float(s[~same & dark].mean()), 3),
          {nm: round(float(S[k][n.index(a), n.index(b)]), 3) for nm, (a, b) in C['hard_pairs'].items()})
# 어두우면 a2로 바꾸기
for q in (.25, .35):
    dk = br < np.quantile(br, q)
    for nm, mask in (('both', dk[:, None] & dk[None, :]), ('either', dk[:, None] | dk[None, :])):
        print(f'switch dark q={q} {nm}', sets(np.where(mask, S['a2'], S['v1']), .611))
print('average v1,a2', sets((S['v1'] + S['a2']) / 2, .611), sets((S['v1'] + S['a2']) / 2, .605))
