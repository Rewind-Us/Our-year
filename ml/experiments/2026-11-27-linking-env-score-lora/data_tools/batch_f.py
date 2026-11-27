import json, random, sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path.home() / 'our-year-train/vt')); from pano_crop import *
from PIL import Image, ImageDraw, ImageFont
FOOD = Path.home() / 'our-year-train/pubeval/raw/pano_food'; FM = json.load(open(FOOD / 'food_meta.json'))
rng = random.Random(7); ids = rng.sample(FM['keep'], 8); S = 230
F_ = ImageFont.truetype('/System/Library/Fonts/AppleSDGothicNeo.ttc', 17)
sheet = Image.new('RGB', (2 * S + 330, 8 * (S + 8)), 'white'); d = ImageDraw.Draw(sheet)
for r, pid in enumerate(ids):
    eq = np.asarray(Image.open(FOOD / 'img' / f'{pid}.jpg').convert('RGB')); y0 = rng.uniform(0, 360); dang = rng.uniform(30, 180)
    for k, yaw in enumerate((y0, y0 + dang)):
        im = crop(eq, yaw, rng.uniform(-10, 10), rng.uniform(60, 80)); sm = np.zeros(im.shape[:2], bool)
        if FM['sky'][pid] > .2:
            q = im.astype(int); sm = ((q[..., 2] > q[..., 0] + 12) & (q[..., 2] > 110)) | ((q[..., 0] > 225) & (q[..., 1] > 225) & (q[..., 2] > 225)); sm[int(.6 * len(sm)):] = False
        pm = np.zeros_like(sm)
        if rng.random() < .7: im, pm, _ = paste(im, rng)
        v = im.copy(); v[sm & ~pm] = (v[sm & ~pm] * .5 + np.array([0, 90, 255]) * .5).astype(np.uint8); v[pm] = (v[pm] * .6 + np.array([255, 0, 0]) * .4).astype(np.uint8)
        sheet.paste(Image.fromarray(v).resize((S, S)), (k * S + (8 if k else 0), r * (S + 8)))
    d.multiline_text((2 * S + 20, r * (S + 8) + 10), f"정답쌍 {r + 1}\n{pid[3:33]}\n{'야외' if FM['sky'][pid] > .2 else '실내'}\n방향 차이 {dang:.0f}도", font=F_, fill='black', spacing=6)
sheet.save('batch_f.jpg', quality=88)
