#!/usr/bin/env python3
"""인물 가림 학습 데이터를 만들어요.

로컬 데이터 폴더(data/private)의 배경과 컷아웃을 합성해 원본, 합성본, 마스크 쌍을 저장해요.
배경과 인물은 train과 synthetic_dev로 나뉘고 서로 겹치지 않아요. 같은 시드는 같은 결과를 만들어요.

사용법: python3 build_occlusion_train.py --private <data/private 경로> --out <출력 폴더>
"""
import argparse, csv, hashlib, json, random
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps, ImageStat

BANDS = {'small': (0.10, 0.25), 'medium': (0.25, 0.40), 'large': (0.40, 0.70)}  # S, 중간, L
PER_BAND = 2
SIDE = 640
MAX_UPSCALE = 2.5
MIN_HEIGHT = 0.45  # 컷아웃 높이 중 화면에 남아야 하는 최소 비율

def rows(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def visible(alpha, x, y, W, H):
    box = (max(0, -x), max(0, -y), min(alpha.width, W - x), min(alpha.height, H - y))
    if box[2] <= box[0] or box[3] <= box[1]:
        return 0
    return sum(alpha.crop(box).histogram()[1:])

def place(cut, W, H, target, ax, top):
    """목표 면적이 되는 크기를 이분 탐색해요. 머리는 화면 안에 두고 아래쪽은 프레임에서 잘려요."""
    lo, hi = 0.02, MAX_UPSCALE
    best = None
    for _ in range(18):
        s = (lo + hi) / 2
        w, h = max(1, round(cut.width * s)), max(1, round(cut.height * s))
        x, y = round(W * ax - w / 2), max(round(H * top), H - h)
        a = cut.getchannel('A').resize((w, h), Image.BILINEAR)
        cov = visible(a, x, y, W, H) / (W * H)
        if best is None or abs(cov - target) < abs(best[0] - target):
            best = (cov, w, h, x, y)
        lo, hi = (s, hi) if cov < target else (lo, s)
    return best

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--private', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--donors', type=Path, default=Path(__file__).with_name('donors.csv'))
    p.add_argument('--seed', type=int, default=20261125)
    a = p.parse_args()
    v1 = {r["background_id"]: r["split"] for r in rows(Path(__file__).with_name("splits.csv"))}
    donors = {'train': [], 'synthetic_dev': []}
    for r in rows(a.donors):
        donors[r['split']].append(r)
    out_rows, skipped, short = [], [], []
    for b in rows(Path(__file__).with_name("backgrounds.csv")):
        bid = b['background_id']
        split = 'synthetic_dev' if v1.get(bid) == 'synthetic_dev' else 'train'
        src = a.private / 'occlusion-v2' / b['original_file']
        bg = ImageOps.exif_transpose(Image.open(src)).convert('RGB')
        bg.thumbnail((SIDE, SIDE), Image.LANCZOS)
        W, H = bg.size
        d = a.out / split
        d.mkdir(parents=True, exist_ok=True)
        orig = f'{split}/{bid}_original.png'
        bg.save(a.out / orig)
        for band, (lo, hi) in BANDS.items():
            seed = int(hashlib.sha256(f'{a.seed}{bid}{band}'.encode()).hexdigest()[:12], 16)
            rng, made = random.Random(seed), 0
            pool = donors[split]
            # 한 명으로 먼저 만들고, 큰 가림에서 인물이 거의 잘리면 두 명을 나란히 놓아요.
            tries = [[d] for d in rng.sample(pool, len(pool))]
            if band == 'large':
                tries += [rng.sample(pool, 2) for _ in range(30)]
            used = set()
            for group in tries:
                if made == PER_BAND:
                    break
                ids = [x['donor_id'] for x in group]
                if used & set(ids):
                    continue
                target = rng.uniform(lo + .02, min(hi, .60) - .02)
                axes = [rng.uniform(.3, .7)] if len(group) == 1 else [rng.uniform(.24, .36), rng.uniform(.64, .76)]
                layer, ok = Image.new('RGBA', (W, H)), True
                for don, ax in zip(group, axes):
                    cut = Image.open(a.private / don['cutout_file']).convert('RGBA')
                    if rng.random() < 0.5:
                        cut = ImageOps.mirror(cut)
                    share = target if len(group) == 1 else target * .55
                    cov, w, h, x, y = place(cut, W, H, share, ax, rng.uniform(.02, .10))
                    side = (max(0, -x) + max(0, x + w - W)) / w
                    if (H - y) / h < MIN_HEIGHT or side > .25:
                        ok = False
                        break
                    one = Image.new('RGBA', (W, H))
                    one.paste(cut.resize((w, h), Image.LANCZOS), (x, y))
                    m1 = one.getchannel('A').point(lambda v: 255 if v > 0 else 0)
                    mf = sum(ImageStat.Stat(one.convert('RGB'), m1).mean)
                    mb = sum(ImageStat.Stat(bg, m1).mean)
                    rgb = ImageEnhance.Brightness(one.convert('RGB')).enhance(max(.85, min(1.15, mb / max(1, mf))))
                    rgb.putalpha(one.getchannel('A'))
                    layer = Image.alpha_composite(layer, rgb)
                alpha = layer.getchannel('A')
                mask = alpha.point(lambda v: 255 if v > 0 else 0)
                n = mask.histogram()[255]
                if not ok or not lo <= n / (W * H) < hi:
                    skipped.append((bid, band, ids))
                    continue
                comp = Image.composite(layer.convert('RGB'), bg, alpha)
                stem = f'{split}/{bid}_{band}{made + 1}'
                comp.save(a.out / f'{stem}_composite.png')
                mask.save(a.out / f'{stem}_mask.png')
                out_rows.append(dict(split=split, scene=b['scene'], background_id=bid, donor_ids='|'.join(ids), people=len(ids), band=band,
                    original_file=orig, composite_file=f'{stem}_composite.png', mask_file=f'{stem}_mask.png',
                    width=W, height=H, mask_pixels=n, total_pixels=W * H, coverage_percent=f'{100 * n / (W * H):.4f}', seed=a.seed))
                used |= set(ids)
                made += 1
            if made < PER_BAND:
                short.append((bid, band, made))
    with open(a.out / 'pairs.csv', 'w', newline='') as f:
        wr = csv.DictWriter(f, out_rows[0].keys()); wr.writeheader(); wr.writerows(out_rows)
    print(json.dumps({'pairs': len(out_rows), 'skipped': len(skipped), 'short': short}))

if __name__ == '__main__':
    main()
