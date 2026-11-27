"""식당, 카페 후보 등장방형 파노라마를 4000px로 받아요. 학습 중인 폴더와 섞이지 않게 pano_food/img에 따로 저장."""
import json, re, time, sys, urllib.parse, urllib.request
from pathlib import Path
O = Path.home() / 'our-year-train/pubeval/raw/pano_food'; (O / 'img').mkdir(parents=True, exist_ok=True)
UA = {'User-Agent': 'research-script/0.1 (place-recognition evaluation)'}
def get(u, raw=False):
    for t in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=60) as r: b = r.read()
            return b if raw else json.loads(b)
        except Exception: time.sleep(5 * (t + 1))
cand = json.load(open(sys.argv[1])); T = [c['title'] for c in cand]; meta = []
for i in range(0, len(T), 40):
    r = get('https://commons.wikimedia.org/w/api.php?' + urllib.parse.urlencode(dict(action='query', format='json', formatversion=2, titles='|'.join(T[i:i + 40]), prop='imageinfo', iiprop='url|extmetadata', iiurlwidth=4000)))
    time.sleep(1)
    for p in (r or {}).get('query', {}).get('pages', []):
        ii = (p.get('imageinfo') or [{}])[0]; em = ii.get('extmetadata', {})
        sid = 'wf_' + re.sub(r'\W+', '_', p['title'][5:])[:80]; out = O / 'img' / f'{sid}.jpg'
        if not out.exists():
            b = get(ii.get('thumburl') or ii['url'], raw=True); time.sleep(1.5)
            if not b: continue
            out.write_bytes(b)
        meta.append(dict(id=sid, title=p['title'], license=em.get('LicenseShortName', {}).get('value', ''), author=re.sub('<[^>]+>', '', em.get('Artist', {}).get('value', ''))[:120], url=ii.get('descriptionurl')))
json.dump(meta, open(O / 'meta.json', 'w'), ensure_ascii=False, indent=0); print('DONE', len(meta))
