"""지금 학습(train_pano_q, --group 1)과 같은 분할, 같은 규칙으로 배치 1개(파노라마 8개 -> 16장)를 그려요."""
import json, random, hashlib, sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path.home() / 'our-year-train/vt')); from pano_crop import *
from PIL import Image, ImageDraw, ImageFont
SKY = json.load(open(PANO.parent / 'sky.json')); GRP = json.load(open(PANO.parent / 'groups.json'))
ids = sorted(x.stem for x in PANO.glob('*.jpg') if SKY.get(x.stem, 0) <= .2)
ids = [i for i in ids if not (i.startswith('wm_') and 'Poly_Haven' in i)]
tr = [i for i in ids if int(hashlib.md5(GRP[i].encode()).hexdigest(), 16) % 10]
print('train panos', len(tr))
rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 3); pids = rng.sample(tr, 8); S = 230
F_ = ImageFont.truetype('/System/Library/Fonts/AppleSDGothicNeo.ttc', 17)
sheet = Image.new('RGB', (2 * S + 330, 8 * (S + 8)), 'white'); d = ImageDraw.Draw(sheet)
for r, pid in enumerate(pids):
    eq = pano(pid); y0 = rng.uniform(0, 360); dang = rng.uniform(30, 180); info = []
    for k, yaw in enumerate((y0, y0 + dang)):
        im = crop(eq, yaw, rng.uniform(-10, 10), rng.uniform(60, 80))
        if rng.random() < .7: im, m, pi = paste(im, rng); cov = pi['cover']
        else: m, cov = np.zeros(im.shape[:2], bool), 0.
        v = im.copy(); v[m] = (v[m] * .6 + np.array([255, 0, 0]) * .4).astype(np.uint8)
        sheet.paste(Image.fromarray(v).resize((S, S)), (k * S + (8 if k else 0), r * (S + 8))); info.append(cov)
    src = 'Poly Haven' if pid.startswith('ph_') else 'Wikimedia'
    d.multiline_text((2 * S + 20, r * (S + 8) + 10), f'정답쌍 {r + 1} (같은 장소)\n{src}\n{pid[3:33]}\n방향 차이 {dang:.0f}도\n사람 A {info[0]*100:.0f}%, B {info[1]*100:.0f}%', font=F_, fill='black', spacing=6)
sheet.save('batch_now.jpg', quality=88)
