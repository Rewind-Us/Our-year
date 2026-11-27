"""등장방형 파노라마에서 휴대폰 사진처럼 원근 투영 조각을 잘라내고 사람을 붙여요."""
import random, numpy as np, cv2
from pathlib import Path
from PIL import Image
PANO = Path.home() / 'our-year-train/pubeval/raw/pano/img'; PEOPLE = sorted(p for p in (Path.home() / 'our-year-train/vt/people').glob('*.png')
                                                                     if (lambda im: im.size[1] > 1.2 * im.size[0])(Image.open(p)))
_cache = {}
def pano(pid):
    if pid not in _cache:
        if len(_cache) > 64: _cache.clear()
        _cache[pid] = np.asarray(Image.open(PANO / f'{pid}.jpg').convert('RGB'))
    return _cache[pid]
def crop(eq, yaw, pitch=0., fov=70., size=322, portrait=False):
    H, W = eq.shape[:2]; w = h = size
    f = .5 * w / np.tan(np.radians(fov) / 2)
    x, y = np.meshgrid(np.arange(w) - w / 2 + .5, np.arange(h) - h / 2 + .5)
    d = np.stack([x, y, np.full_like(x, f)], -1); d /= np.linalg.norm(d, axis=-1, keepdims=True)
    cp, sp, cy, sy = np.cos(np.radians(pitch)), np.sin(np.radians(pitch)), np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]]); Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    d = d @ (Ry @ Rx).T
    lon = np.arctan2(d[..., 0], d[..., 2]); lat = np.arcsin(np.clip(d[..., 1], -1, 1))
    u = ((lon / (2 * np.pi) + .5) * W).astype(np.float32); v = ((lat / np.pi + .5) * H).astype(np.float32)
    return cv2.remap(eq, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
# 가림 구간은 data/private/occlusion-v2 기준(화면 면적 %), 40~70%는 촬영표 L 컷 목표
BANDS = {'wide': (5, 12), 'medium': (12, 25), 'close': (25, 40), 'large': (40, 70)}
BAND_P = {'wide': .2, 'medium': .3, 'close': .3, 'large': .2}
POS = {'left': 1 / 6, 'center': .5, 'right': 5 / 6}
def paste(im, rng, band=None, pos=None):
    """사람을 구간 면적과 좌, 중, 우 위치에 맞춰 붙이고 (이미지, 마스크, 정보) 반환. 화면 밖으로 잘린 부분은 면적에서 빠져요."""
    im = Image.fromarray(im); S = im.size[0]
    band = band or rng.choices(list(BAND_P), list(BAND_P.values()))[0]; pos = pos or rng.choice(list(POS))
    target = rng.uniform(*BANDS[band]) / 100; p = Image.open(rng.choice(PEOPLE)); a0 = np.asarray(p.split()[3]) > 127
    cx = POS[pos] * S + rng.uniform(-.06, .06) * S; top = rng.uniform(.03, .15)
    lo, hi = .1, 4.
    for _ in range(18):  # 실제 보이는 면적이 목표에 맞도록 크기를 이분 탐색
        h = int(S * (lo + hi) / 2); w = max(1, int(p.size[0] * h / p.size[1]))
        x = int(min(max(cx - w / 2, -.25 * w), S - .75 * w))  # 몸의 3/4 이상은 화면 안
        y = S - h + int(h * .1) if h < .9 * S else int(top * S)  # 가까우면 머리를 위에 두고 아래가 잘리게
        m = Image.new('L', (S, S)); m.paste(Image.fromarray((a0 * 255).astype(np.uint8)).resize((w, h)), (x, y))
        cov = (np.asarray(m) > 127).mean()
        if cov < target: lo = (lo + hi) / 2
        else: hi = (lo + hi) / 2
    q = p.resize((w, h), Image.BICUBIC); im.paste(q, (x, y), q)
    mask = np.asarray(m) > 127
    return np.asarray(im), mask, dict(band=band, pos=pos, cover=round(float(mask.mean()), 3))
def pair(pid, ang, rng, person_p=.7):
    eq = pano(pid); y0 = rng.uniform(0, 360); out = []
    for yaw in (y0, y0 + ang):
        im = crop(eq, yaw, rng.uniform(-10, 10), rng.uniform(60, 80))
        if rng.random() < person_p: im, m, info = paste(im, rng)
        else: m, info = np.zeros(im.shape[:2], bool), dict(band='none', pos='-', cover=0.)
        out.append((im, m, info))
    return out
