"""목적 맞춤 LoRA: 같은 파노라마의 30~180도 조각 쌍은 '같은 환경 점수'(사람 조각 제외 chamfer)가 높게, 다른 파노라마는 낮게.
평가: 학습에 안 쓴 파노라마 10%(각도별), ETH3D(각도별, 다른 장소 오탐 1% 기준선)."""
import argparse, sys, json, time, random, hashlib, itertools, numpy as np, torch
from common import *
from pano_crop import PANO, pano, crop, paste
p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, required=True); p.add_argument('--steps', type=int, default=1500); p.add_argument('--batch', type=int, default=8)
p.add_argument('--blocks', type=int, default=4); p.add_argument('--rank', type=int, default=8); p.add_argument('--lr', type=float, default=1e-4)
p.add_argument('--tau', type=float, default=.03); p.add_argument('--kd', type=float, default=1.); p.add_argument('--eval_every', type=int, default=500); p.add_argument('--resume', type=Path); p.add_argument('--start', type=int, default=0); p.add_argument('--hard', type=float, default=0.); p.add_argument('--gc', type=int, default=0)
p.add_argument('--seed', type=int, default=0); p.add_argument('--group', type=int, default=0)  # 1: 장소 그룹 단위 분할 + Wikimedia의 Poly Haven 복제본 제외 + 같은 그룹은 오답에서 제외
p.add_argument('--food', type=int, default=0); p.add_argument('--outdoor', type=int, default=0); p.add_argument('--sky', type=int, default=0)
p.add_argument('--backbone', default='megaloc'); p.add_argument('--head', type=int, default=0)  # dinov2: 원본 DINOv2 가중치로 교체 / head N: 백본 고정, 패치 위 변환층(은닉 N)만 학습
p.add_argument('--queue', type=int, default=0); p.add_argument('--queue_start', type=int, default=200)  # 이전 배치 조각을 오답 후보로 쌓는 큐 크기
a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True); random.seed(a.seed); torch.manual_seed(a.seed)
if dev == 'mps': torch.mps.set_per_process_memory_fraction(.6)
import subprocess; subprocess.run([sys.executable, '-I', str(VT / 'sky_filter.py')], capture_output=True)
SKY = json.load(open(PANO.parent / 'sky.json'))
ids = sorted(x.stem for x in PANO.glob('*.jpg') if a.outdoor or SKY.get(x.stem, 0) <= .2)
FOOD = Path.home() / 'our-year-train/pubeval/raw/pano_food'; FM = json.load(open(FOOD / 'food_meta.json')) if a.food else dict(keep=[], groups={}, sky={})
FM['keep'] = [i for i in FM['keep'] if a.outdoor or FM['sky'][i] <= .2]; SKY.update(FM['sky'])
import pano_crop
def pano(pid):
    if pid not in pano_crop._cache:
        if len(pano_crop._cache) > 64: pano_crop._cache.clear()
        from PIL import Image as _I; pano_crop._cache[pid] = np.asarray(_I.open((FOOD / 'img' if pid.startswith('wf_') else PANO) / f'{pid}.jpg').convert('RGB'))
    return pano_crop._cache[pid]  # 야외(하늘 넓음) Wikimedia 제외
print('excluded outdoor', sum(v > .2 for v in SKY.values()), flush=True); split = lambda i: int(hashlib.md5(i.encode()).hexdigest(), 16) % 10
GRP = {i: i for i in ids}
if a.group:
    GRP = json.load(open(PANO.parent / 'groups.json'))
    dup = [i for i in ids if i.startswith('wm_') and 'Poly_Haven' in i]; ids = [i for i in ids if i not in dup]
    GRP = {**{i: GRP.get(i, i) for i in ids}, **FM['groups']}; ids = ids + FM['keep']
    print('dropped Poly Haven copies', len(dup), 'food added', len(FM['keep']), flush=True); split = lambda i: int(hashlib.md5(GRP[i].encode()).hexdigest(), 16) % 10
tr, te = [i for i in ids if split(i)], [i for i in ids if not split(i)]
print('panos train', len(tr), 'test', len(te), flush=True)
net = megaloc(a.blocks, a.rank); loras = [m for m in net.modules() if isinstance(m, LoRA)]
if a.backbone == 'dinov2':
    import models; r = net.m.backbone.load_state_dict(models.dino(True).state_dict(), strict=False)
    print('dinov2 weights loaded, missing', len(r.missing_keys), r.missing_keys[:5], 'unexpected', len(r.unexpected_keys), flush=True); net.to(dev)
class PHead(nn.Module):  # 패치마다 같은 작은 변환(잔차). 출력층 0 초기화라 처음엔 원래 특징과 동일
    def __init__(s, d=768, h=256):
        super().__init__(); s.a, s.b, s.on = nn.Linear(d, h), nn.Linear(h, d), True; nn.init.zeros_(s.b.weight); nn.init.zeros_(s.b.bias)
    def forward(s, f): return f + s.b(F.gelu(s.a(f))) if s.on else f
head = PHead(768, a.head).to(dev) if a.head else None
if a.gc:  # 학습 블록 활성값을 저장하지 않고 역전파 때 다시 계산(메모리 절약, 결과 동일)
    from torch.utils.checkpoint import checkpoint
    for blk in net.m.backbone.blocks[-a.blocks:]:
        blk.forward = (lambda f: (lambda x: checkpoint(f, x, use_reentrant=False) if torch.is_grad_enabled() else f(x)))(blk.forward)
def lora_on(on):
    for m in loras: m.scale = 2. if on else 0.
    if head: head.on = on
G = 23
def tens(im): return (torch.from_numpy(im.astype(np.float32) / 255).permute(2, 0, 1) - MEAN) / STD
def pmask(m):  # 322 마스크 -> 23x23 조각 중 사람이 절반 넘는 조각
    return torch.from_numpy(m[:G * 14, :G * 14].reshape(G, 14, G, 14).mean((1, 3)) > .5).flatten()
def fwd(x):
    pf, cls = net.m.backbone(x); g = F.normalize(net.m.aggregator((pf, cls)), dim=1)
    t = pf.flatten(2).transpose(1, 2)
    return F.normalize(head(t) if head else t, dim=2), g
def chamfer_mat(A, B, ma, mb):
    """A,B: (n,P,D), ma,mb: (n,P) 사람 조각 True. 반환 (n,n) 같은 환경 점수."""
    S = torch.einsum('ipd,jqd->ijpq', A, B)
    S = S.masked_fill(mb[None, :, None, :], -2).masked_fill(ma[:, None, :, None], -2)
    va, vb = (~ma).float(), (~mb).float()
    ab = (S.max(3).values * va[:, None]).sum(2) / va.sum(1)[:, None].clamp(min=1)
    ba = (S.max(2).values.clamp(min=-1) * vb[None]).sum(2) / vb.sum(1)[None].clamp(min=1)
    return .5 * (ab + ba)
def batch(pids, rng, angs=(30, 180), person_p=.7):
    X, M = [], []
    for pid in pids:
        eq = pano(pid); y0 = rng.uniform(0, 360); d = rng.uniform(*angs)
        for yaw in (y0, y0 + d):
            im = crop(eq, yaw, rng.uniform(-10, 10), rng.uniform(60, 80))
            m = np.zeros(im.shape[:2], bool)
            if a.sky and SKY.get(pid, 0) > .2:  # 야외: 위쪽 60% 안의 파란 하늘, 밝은 흰 하늘 픽셀은 점수에서 제외
                q = im.astype(int); r_, g_, b_ = q[..., 0], q[..., 1], q[..., 2]
                m = (((b_ > r_ + 12) & (b_ > 110)) | ((r_ > 225) & (g_ > 225) & (b_ > 225))); m[int(.6 * len(m)):] = False
            if rng.random() < person_p: im, mp, _ = paste(im, rng); m = m | mp
            X.append(tens(im)); M.append(pmask(m))
    return torch.stack(X), torch.stack(M)
# 고정 평가 세트
erng = random.Random(123); EV = []
for pid in te:
    for ang in (30, 60, 90, 120, 180):
        x, m = batch([pid], erng, (ang - 1e-3, ang)); EV.append((pid, ang, x, m))
@torch.no_grad()
def eval_pano():
    net.eval(); F_, M_ = [], []
    for pid, ang, x, m in EV:
        f, _ = fwd(x.to(dev)); F_.append(f.half().cpu()); M_.append(m)
    pos = {}; neg = []
    for k, (pid, ang, _, _) in enumerate(EV):
        s = chamfer_mat(F_[k][:1].float().to(dev), F_[k][1:].float().to(dev), M_[k][:1].to(dev), M_[k][1:].to(dev))[0, 0].item(); pos.setdefault(ang, []).append(s)
    r = random.Random(7)
    for _ in range(1500):
        i, j = r.sample(range(len(EV)), 2)
        if GRP[EV[i][0]] == GRP[EV[j][0]]: continue  # 같은 장소 그룹은 오답 쌍에서 제외
        neg.append(chamfer_mat(F_[i][:1].float().to(dev), F_[j][1:].float().to(dev), M_[i][:1].to(dev), M_[j][1:].to(dev))[0, 0].item())
    t = float(np.quantile(neg, .99))
    return dict(thr=round(t, 3), **{str(k): round(float((np.array(v) > t).mean()), 3) for k, v in sorted(pos.items())})
# ETH3D
exec(open(VT / 'eth3d_exp.py').read().split('people = ')[0].split('scenes = ')[0])
scenes = sorted(q for q in R.iterdir() if (q / 'dslr_calibration_jpg').exists()); items = []
for sd in scenes:
    for it in load_scene(sd): it['scene'] = sd.name; it['place'] = place(sd.name); items.append(it)
EX = torch.stack([img(it['path']) for it in items]); epos = json.load(open(VT / 'eth3d/pos_pairs.json'))
nr = np.random.default_rng(0); eneg = []
while len(eneg) < 20000:
    i, j = nr.integers(0, len(items), 2)
    if items[i]['place'] != items[j]['place']: eneg.append((int(i), int(j)))
@torch.no_grad()
def eval_eth():
    net.eval(); P = torch.cat([fwd(EX[i:i + 16].to(dev))[0].half().cpu() for i in range(0, len(EX), 16)])
    def sc(pairs):
        out = []
        for k in range(0, len(pairs), 64):
            I, J = zip(*pairs[k:k + 64]); A, B = P[list(I)].float().to(dev), P[list(J)].float().to(dev)
            s = A @ B.transpose(1, 2); out.append((.5 * (s.max(2).values.mean(1) + s.max(1).values.mean(1))).cpu())
        return torch.cat(out).numpy()
    sn = sc(eneg); t = float(np.quantile(sn, .99)); sp = sc([(x['i'], x['j']) for x in epos]); ang = np.array([x['ang'] for x in epos])
    return dict(thr=round(t, 3), **{f'{lo}-{hi}': round(float((sp[(ang >= lo) & (ang < hi)] > t).mean()), 3) for lo, hi in ((0, 30), (30, 60), (60, 90), (90, 120), (120, 181))})
import gc
def evaluate(step):
    r = dict(step=step, pano=eval_pano(), eth3d=eval_eth()); print(json.dumps(r), flush=True)
    gc.collect(); torch.mps.empty_cache() if dev == 'mps' else None; return r
if a.resume: print('resume', net.load_state_dict(torch.load(a.resume, map_location=dev), strict=False).unexpected_keys, flush=True)
hist = [evaluate(a.start)]
NB = None
if a.hard:  # 학습 파노라마마다 4방향 조각의 전역 특징 평균으로 가까운 이웃 20개
    with torch.no_grad():
        emb = []
        for pid in tr:
            x = torch.stack([tens(crop(pano(pid), y, 0, 70)) for y in (0, 90, 180, 270)]).to(dev)
            emb.append(F.normalize(fwd(x)[1].mean(0), dim=0).cpu())
        E_ = torch.stack(emb); Sim = E_ @ E_.T; Sim.fill_diagonal_(-9); NB = Sim.topk(20, dim=1).indices.tolist()
    gc.collect(); torch.mps.empty_cache() if dev == 'mps' else None
    print('hard neighbors ready', flush=True)
def pick(rng):
    if NB and rng.random() < a.hard:
        i = rng.randrange(len(tr)); return [tr[i]] + [tr[j] for j in rng.sample(NB[i], a.batch - 1)]
    return rng.sample(tr, a.batch)
params = [x for m in loras for x in (m.a.weight, m.b.weight)] + (list(head.parameters()) if head else [])
print('trainable params', sum(x.numel() for x in params), flush=True)
for x in params: x.requires_grad_(True)
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 50) * .5 * (1 + np.cos(np.pi * s / max(a.steps, 1))))
for _ in range(a.start): sched.step()
rng = random.Random(1 + a.start + 1000 * a.seed); t0 = time.time()
Qf, Qm, Qg = None, None, []
def neg_mask(ga, gb):  # 같은 그룹(다른 파노라마라도 같은 장소)은 오답으로 쓰지 않음, 대각선(정답)은 남김
    return torch.tensor([[x == y for y in gb] for x in ga], device=dev)
for step in range(a.start + 1, a.steps + 1):
    net.train(); pids = pick(rng); bg = [GRP[i] for i in pids]; x, m = batch(pids, rng); x, m = x.to(dev), m.to(dev)
    need_kd = a.kd > 0 or step % 20 == 0  # kd=0이면 손실에 안 들어가므로 기록용으로 20스텝마다만 계산
    if need_kd:
        with torch.no_grad(): lora_on(False); Tp, Tg = fwd(x); lora_on(True)
    Pf, Gf = fwd(x)
    C = chamfer_mat(Pf[0::2], Pf[1::2], m[0::2], m[1::2]) / a.tau; y = torch.arange(a.batch, device=dev)
    if a.group: C = C.masked_fill(neg_mask(bg, bg) & ~torch.eye(a.batch, dtype=torch.bool, device=dev), -1e4)
    Cr = C
    if a.queue and step > a.queue_start and Qg:
        CQ = torch.cat([chamfer_mat(Pf[0::2], Qf[k:k + 8].float(), m[0::2], Qm[k:k + 8]) for k in range(0, len(Qg), 8)], 1) / a.tau
        Cr = torch.cat([C, CQ.masked_fill(neg_mask(bg, Qg), -1e4)], 1)
    l_nce = .5 * (F.cross_entropy(Cr, y) + F.cross_entropy(C.T, y))
    l_kd = ((1 - (Pf * Tp).sum(2)).mean() + (1 - (Gf * Tg).sum(1)).mean()) if need_kd else torch.zeros((), device=dev)
    if a.queue:
        nf, nm = Pf[1::2].detach().half(), m[1::2]
        Qf = nf if Qf is None else torch.cat([nf, Qf])[:a.queue]; Qm = nm if Qm is None else torch.cat([nm, Qm])[:a.queue]; Qg = (bg + Qg)[:a.queue]
    loss = l_nce + a.kd * l_kd; opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if step % 20 == 0: print(f'step {step} nce {l_nce.item():.3f} kd {l_kd.item():.4f} {(time.time()-t0)/(step-a.start):.2f}s/step', flush=True)
    if step % a.eval_every == 0 or step == a.steps:
        torch.save({**{k: v.cpu() for k, v in net.state_dict().items() if '.qkv.a.' in k or '.qkv.b.' in k}, **({'head.' + k: v.cpu() for k, v in head.state_dict().items()} if head else {})}, a.out / f'step{step}.ckpt'); hist.append(evaluate(step))
json.dump(dict(args={k: str(v) for k, v in vars(a).items()}, hist=hist, sec=round(time.time() - t0)), open(a.out / 'report.json', 'w'), indent=1)
print('DONE', flush=True)
