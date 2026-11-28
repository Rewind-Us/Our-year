"""train_pano_f.py -> train_pano_a.py: 학습 조각에만 휴대폰 사진 변형(--aug 1~3)을 넣는다. 평가 세트(EV)는 그대로."""
from pathlib import Path
src = Path.home() / 'our-year-train/vt/train_pano_f.py'; s = src.read_text()
s = s.replace("p.add_argument('--food', type=int, default=0);",
              "p.add_argument('--aug', type=int, default=0); p.add_argument('--food', type=int, default=0);", 1)
AUG = '''
import cv2 as _cv2
def crop_wh(eq, yaw, pitch, hfov, w, h):
    """가로 화각 hfov, 크기 w x h 로 자르기."""
    H, W = eq.shape[:2]; f = .5 * w / np.tan(np.radians(hfov) / 2)
    x, y = np.meshgrid(np.arange(w) - w / 2 + .5, np.arange(h) - h / 2 + .5)
    d = np.stack([x, y, np.full_like(x, f)], -1); d /= np.linalg.norm(d, axis=-1, keepdims=True)
    cp, sp, cy, sy = np.cos(np.radians(pitch)), np.sin(np.radians(pitch)), np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]]); Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    d = d @ (Ry @ Rx).T
    lon = np.arctan2(d[..., 0], d[..., 2]); lat = np.arcsin(np.clip(d[..., 1], -1, 1))
    u = ((lon / (2 * np.pi) + .5) * W).astype(np.float32); v = ((lat / np.pi + .5) * H).astype(np.float32)
    return _cv2.remap(eq, u, v, _cv2.INTER_LINEAR, borderMode=_cv2.BORDER_WRAP)
def phone_view(eq, yaw, rng, lvl):
    """lvl 1: 세로 3:4(80%) 또는 가로 4:3으로 자른 뒤 322x322로 눌러 실제 입력과 맞춤.
    lvl 3: 화각 최대 110도, 기울기 최대 +-35도."""
    hf = rng.uniform(50, 75) if lvl < 3 else (rng.uniform(50, 75) if rng.random() < .6 else rng.uniform(75, 110))
    pt = rng.uniform(-10, 10) if lvl < 3 else (rng.uniform(-10, 10) if rng.random() < .6 else rng.choice((-1, 1)) * rng.uniform(10, 35))
    w, h = (242, 322) if rng.random() < .8 else (322, 242)
    im = crop_wh(eq, yaw, pt, hf, w, h)
    return np.asarray(Image.fromarray(im).resize((322, 322), Image.BICUBIC))
def photometric(im, rng):
    """밝기와 감마(어두운 실내), 대비, 색온도, 노이즈, 흐림, JPEG 압축."""
    x = im.astype(np.float32) / 255
    if rng.random() < .8:
        x = np.clip(x, 0, 1) ** rng.uniform(.8, 1.8) * rng.uniform(.45, 1.15)
    if rng.random() < .5:
        mu = x.mean(); x = (x - mu) * rng.uniform(.7, 1.3) + mu
    if rng.random() < .6:
        t = rng.uniform(-.15, .15); x = x * np.array([1 + t, 1, 1 - t], np.float32)
    if rng.random() < .4:
        x = x + np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(.005, .03), x.shape).astype(np.float32)
    x = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    if rng.random() < .3:
        k = rng.choice((3, 5)); x = _cv2.GaussianBlur(x, (k, k), 0)
    if rng.random() < .5:
        ok, b = _cv2.imencode('.jpg', x[..., ::-1], [_cv2.IMWRITE_JPEG_QUALITY, rng.randint(30, 90)]); x = _cv2.imdecode(b, 1)[..., ::-1].copy()
    return x
import pano_crop as _pc
if a.aug >= 3:
    _pc.BANDS['xl'] = (70, 80); _pc.BAND_P.update({'wide': .18, 'medium': .28, 'close': .28, 'large': .18, 'xl': .08})
'''
old_batch = s[s.index('def batch(pids'):s.index('# 고정 평가 세트')]
new_batch = old_batch.replace('def batch(pids, rng, angs=(30, 180), person_p=.7):', 'def batch(pids, rng, angs=(30, 180), person_p=.7, aug=0):')
new_batch = new_batch.replace("            im = crop(eq, yaw, rng.uniform(-10, 10), rng.uniform(60, 80))",
"            im = crop(eq, yaw, rng.uniform(-10, 10), rng.uniform(60, 80)) if not aug else phone_view(eq, yaw, rng, aug)\n"
"            if aug >= 2: im = photometric(im, rng)\n"
"            if aug >= 3 and rng.random() < .5: im = np.ascontiguousarray(im[:, ::-1])")
assert new_batch != old_batch
s = s.replace(old_batch, AUG + new_batch, 1)
old = "x, m = batch(pids, rng); x, m = x.to(dev), m.to(dev)"
assert old in s; s = s.replace(old, "x, m = batch(pids, rng, aug=a.aug); x, m = x.to(dev), m.to(dev)", 1)
(src.parent / 'train_pano_a.py').write_text(s); print('ok')
