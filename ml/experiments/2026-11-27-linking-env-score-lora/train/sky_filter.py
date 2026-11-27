"""Wikimedia 파노라마 중 하늘이 넓게 보이는 야외 사진을 골라내요(위쪽 25% 영역의 하늘색 비율)."""
import json, numpy as np
from PIL import Image
from pathlib import Path
P = Path.home() / 'our-year-train/pubeval/raw/pano/img'; out = {}
for f in sorted(P.glob('wm_*.jpg')):
    a = np.asarray(Image.open(f).convert('RGB').resize((512, 256))).astype(int)[:64]
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    sky = ((b > r + 12) & (b > 110)) | ((r > 225) & (g > 225) & (b > 225))
    out[f.stem] = round(float(sky.mean()), 3)
json.dump(out, open(P.parent / 'sky.json', 'w'))
v = sorted(out.items(), key=lambda x: -x[1]); print(len(v), 'wm'); print('sky>0.25:', sum(x > .25 for _, x in v))
for k, x in v[:8] + v[len(v)//2:len(v)//2+3]: print(x, k[:60])
