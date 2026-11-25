#!/usr/bin/env python3
"""MegaLoc 후반 블록 qkv에 LoRA를 붙여 가림 증류를 시범 학습해요.

L = 0.5 * (1 - cos(S(occ), T(clean))) + 0.5 * (1 - cos(S(clean), T(clean)))
사용법: python3 train_lora.py --data <데이터> --model <megaloc 폴더> --out <출력> [--max-steps 10]
"""
import argparse, collections, csv, json, random, re, subprocess, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from PIL import Image
from safetensors.torch import load_file

MEAN, STD = torch.tensor([.485, .456, .406]).view(3, 1, 1), torch.tensor([.229, .224, .225]).view(3, 1, 1)

class LoRA(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base, self.scale = base, alpha / rank
        self.a, self.b = nn.Linear(base.in_features, rank, bias=False), nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=5 ** .5); nn.init.zeros_(self.b.weight)
    def forward(self, x):
        return self.base(x) + self.scale * self.b(self.a(x))

def image(path, mask=None):
    im = np.asarray(Image.open(path).convert('RGB').resize((322, 322), Image.BICUBIC), dtype=np.float32) / 255
    x = (torch.from_numpy(im.copy()).permute(2, 0, 1) - MEAN) / STD
    if mask is not None:
        m = np.asarray(Image.open(mask).convert('L').resize((322, 322), Image.NEAREST)) > 0
        x[:, torch.from_numpy(m.copy())] = 0
    return x

def swap_mb():
    if sys.platform != 'darwin': return 0.
    o = subprocess.run(['sysctl', '-n', 'vm.swapusage'], capture_output=True, text=True).stdout
    return float(re.search(r'used = ([\d.]+)M', o).group(1))

def main():
    p = argparse.ArgumentParser()
    for k in ('data', 'model', 'out'): p.add_argument('--' + k, type=Path, required=True)
    p.add_argument('--epochs', type=int, default=5); p.add_argument('--max-steps', type=int, default=0)
    p.add_argument('--batch', type=int, default=2); p.add_argument('--rank', type=int, default=8)
    p.add_argument('--blocks', type=int, default=4); p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--swap-limit-mb', type=float, default=1024); p.add_argument('--mem-fraction', type=float, default=.35)
    p.add_argument('--seed', type=int, default=0); p.add_argument('--resume', type=Path)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed); torch.manual_seed(a.seed)
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    if dev == 'mps': torch.mps.set_per_process_memory_fraction(a.mem_fraction)
    swap0 = swap_mb()
    sys.path.insert(0, str(a.model))
    from megaloc_model import MegaLoc
    net = MegaLoc(); net.load_state_dict(load_file(a.model / 'model.safetensors')); net.eval().requires_grad_(False)
    net.to(dev)
    rows = list(csv.DictReader(open(a.data / 'pairs.csv', encoding='utf-8')))
    train, val = [[r for r in rows if r['split'] == s] for s in ('train', 'synthetic_dev')]
    cleans = sorted({r['original_file'] for r in rows})

    @torch.no_grad()
    def embed(items):
        out = []
        for i in range(0, len(items), 4):
            out.append(net(torch.stack([image(a.data / f, m and a.data / m) for f, m in items[i:i + 4]]).to(dev)).cpu())
            if dev == 'mps': torch.mps.empty_cache()
        return torch.cat(out)

    teacher = dict(zip(cleans, embed([(c, None) for c in cleans])))
    gallery = torch.stack([teacher[c] for c in cleans])

    def evaluate(masked=False):
        """갤러리는 원본 교사 특징 57장이에요. 원본이 질의의 배경과 같은 사진이라 실제 R@1보다 쉬워요."""
        q = embed([(r['composite_file'], r['mask_file'] if masked else None) for r in val])
        t = torch.stack([teacher[r['original_file']] for r in val])
        sim, hit = (q * t).sum(1), ((q @ gallery.T).argmax(1) == torch.tensor([cleans.index(r['original_file']) for r in val]))
        d = {}
        for b in ('small', 'medium', 'large'):
            ix = [i for i, r in enumerate(val) if r['band'] == b]
            d[b] = dict(n=len(ix), cos=round(sim[ix].mean().item(), 4), r1=int(hit[ix].sum()))
        vc = sorted({r['original_file'] for r in val})
        c = embed([(x, None) for x in vc])
        d['clean_cos'] = round((c * torch.stack([teacher[x] for x in vc])).sum(1).mean().item(), 4)
        return d

    report = dict(args={k: str(v) for k, v in vars(a).items()}, torch=torch.__version__, device=dev,
                  E0=evaluate(), E1_mask=evaluate(True))
    print(json.dumps(report), flush=True)
    for blk in net.backbone.blocks[-a.blocks:]:
        blk.attn.qkv = LoRA(blk.attn.qkv, a.rank, 2 * a.rank).to(dev)
    names = [n for n, x in net.named_parameters() if x.requires_grad]
    params = [x for x in net.parameters() if x.requires_grad]
    print('trainable', sum(x.numel() for x in params), names, flush=True)
    if a.resume: net.load_state_dict(torch.load(a.resume, map_location=dev), strict=False)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=.01)
    save = lambda n: torch.save({k: v.cpu() for k, v in net.state_dict().items() if '.qkv.a.' in k or '.qkv.b.' in k}, a.out / n)
    log, step, stop, t0, peak, swap_peak = [], 0, None, time.time(), 0, swap0
    for ep in range(a.epochs):
        order = random.sample(train, len(train)); acc = collections.defaultdict(list)
        for i in range(0, len(order) - a.batch + 1, a.batch):
            rs = order[i:i + a.batch]
            t = torch.stack([teacher[r['original_file']] for r in rs]).to(dev)
            opt.zero_grad()
            for k in ('composite_file', 'original_file'):
                s = net(torch.stack([image(a.data / r[k]) for r in rs]).to(dev))
                l = .5 * (1 - (s * t).sum(1)).mean(); l.backward()
                acc[k].append(2 * l.item())
            opt.step(); step += 1
            if dev == 'mps': peak = max(peak, torch.mps.driver_allocated_memory())
            sw = swap_mb(); swap_peak = max(swap_peak, sw)
            if step % 10 == 0:
                print(f'step {step} ep {ep} occ {np.mean(acc["composite_file"][-10:]):.4f} clean {np.mean(acc["original_file"][-10:]):.4f} '
                      f'{(time.time() - t0) / step:.2f}s/step mps {peak / 1e6:.0f}MB swap+{sw - swap0:.0f}MB', flush=True)
            if step % 50 == 0: save('last.ckpt')
            if sw - swap0 > a.swap_limit_mb: stop = 'swap limit'
            if a.max_steps and step >= a.max_steps: stop = 'max steps'
            if stop: break
        log.append(dict(epoch=ep, occ=float(np.mean(acc['composite_file'])), clean=float(np.mean(acc['original_file']))))
        if not stop: log[-1]['eval'] = evaluate(); print(json.dumps(log[-1]), flush=True)
        if stop: break
    save('last.ckpt')
    report.update(log=log, steps=step, stop=stop, seconds=round(time.time() - t0), peak_mps_mb=round(peak / 1e6),
                  swap_start_mb=swap0, swap_peak_mb=swap_peak)
    if not stop: report['E3'] = evaluate()
    (a.out / 'report.json').write_text(json.dumps(report, indent=1))
    print('DONE', json.dumps({k: report[k] for k in ('steps', 'stop', 'seconds', 'peak_mps_mb', 'swap_peak_mb')}), flush=True)

if __name__ == '__main__':
    main()
