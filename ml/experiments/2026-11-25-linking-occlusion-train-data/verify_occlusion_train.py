#!/usr/bin/env python3
"""합성 데이터의 크기, 마스크 밖 픽셀, 가림 비율, 분할 겹침을 확인해요. 문제가 있으면 종료 코드 1이에요.

사용법: python3 verify_occlusion_train.py <데이터 폴더>
"""
import collections, csv, sys
from pathlib import Path
from PIL import Image, ImageChops
from build_occlusion_train import BANDS

root = Path(sys.argv[1])
rows = list(csv.DictReader(open(root / 'pairs.csv', encoding='utf-8')))
errors = []
for r in rows:
    o, c = (Image.open(root / r[k]).convert('RGB') for k in ('original_file', 'composite_file'))
    m = Image.open(root / r['mask_file']).convert('L')
    if not (o.size == c.size == m.size):
        errors.append((r['composite_file'], 'size')); continue
    h = m.histogram()
    if sum(h[1:255]): errors.append((r['mask_file'], 'not binary'))
    if h[255] != int(r['mask_pixels']): errors.append((r['mask_file'], 'pixel count'))
    lo, hi = BANDS[r['band']]
    if not lo <= h[255] / sum(h) < hi: errors.append((r['mask_file'], 'band'))
    diff = ImageChops.difference(o, c); diff.paste((0, 0, 0), mask=m)
    if diff.getbbox(): errors.append((r['composite_file'], 'outside mask changed'))
for k in ('background_id', 'donor_ids'):
    s = [{v for r in rows if r['split'] == x for v in r[k].split('|')} for x in ('train', 'synthetic_dev')]
    print(k, len(s[0]), len(s[1]))
    if s[0] & s[1]: errors.append((k, 'split overlap'))
print(len(rows), collections.Counter((r['split'], r['band']) for r in rows))
print('donors', sorted(collections.Counter(d for r in rows if r['split'] == 'train' for d in r['donor_ids'].split('|')).values()))
print('people', collections.Counter((r['band'], r['people']) for r in rows))
print('errors', errors)
sys.exit(bool(errors))
