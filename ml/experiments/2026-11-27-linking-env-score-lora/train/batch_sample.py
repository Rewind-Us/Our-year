"""학습 배치 1개(파노라마 8개 -> 16장)를 학습과 같은 규칙으로 만들고 정답 표시와 함께 저장해요."""
import json, random, hashlib, numpy as np
from PIL import Image, ImageDraw
import sys; sys.path.insert(0, '.'); from pano_crop import *
meta = {m['id']: m for m in json.load(open(PANO.parent / 'meta.json'))} if (PANO.parent / 'meta.json').exists() else {}
ids = sorted(p.stem for p in PANO.glob('*.jpg')); tr = [i for i in ids if int(hashlib.md5(i.encode()).hexdigest(), 16) % 10]
rng = random.Random(1); pids = rng.sample(tr, 8); S = 190; rows = []
sheet = Image.new('RGB', (2 * S + 520, 8 * (S + 6)), 'white'); d = ImageDraw.Draw(sheet)
for r, pid in enumerate(pids):
    eq = pano(pid); y0 = rng.uniform(0, 360); dang = rng.uniform(30, 180); info = []
    for k, yaw in enumerate((y0, y0 + dang)):
        pitch, fov = rng.uniform(-10, 10), rng.uniform(60, 80); im = crop(eq, yaw, pitch, fov)
        if rng.random() < .7: im, m, pi = paste(im, rng)
        else: m, pi = np.zeros(im.shape[:2], bool), dict(band='none', pos='-', cover=0.)
        v = im.copy(); v[m] = (v[m] * .55 + np.array([255, 0, 0]) * .45).astype(np.uint8)
        sheet.paste(Image.fromarray(v).resize((S, S)), (k * S, r * (S + 6)))
        info.append(dict(yaw=round(yaw % 360), pitch=round(pitch, 1), fov=round(fov), **pi))
    src = 'Poly Haven, CC0' if pid.startswith('ph_') else f"Wikimedia, {meta.get(pid, {}).get('license', '?')}"
    rows.append(dict(pair=r, pano=pid, source=src, yaw_diff=round(dang), A=info[0], B=info[1]))
    t = (f"pair {r}  label: SAME (positive)\n{pid[:40]}\n{src}\nview diff {dang:.0f} deg\n"
         f"A: person {info[0]['band']} {info[0]['pos']} {info[0]['cover']*100:.0f}%  fov {info[0]['fov']}\n"
         f"B: person {info[1]['band']} {info[1]['pos']} {info[1]['cover']*100:.0f}%  fov {info[1]['fov']}\n"
         f"negatives: B of the other 7 pairs")
    d.multiline_text((2 * S + 10, r * (S + 6) + 8), t, fill='black', spacing=4)
sheet.save('batch_sample.jpg', quality=85); json.dump(rows, open('batch_sample.json', 'w'), indent=1, ensure_ascii=False)
