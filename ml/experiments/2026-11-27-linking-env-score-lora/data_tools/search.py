import json, time, urllib.parse, urllib.request, re
UA = {'User-Agent': 'research-script/0.1 (place-recognition evaluation)'}
def api(**p):
    p.update(format='json', formatversion=2)
    for t in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request('https://commons.wikimedia.org/w/api.php?' + urllib.parse.urlencode(p), headers=UA), timeout=30) as r: return json.load(r)
        except Exception: time.sleep(3 * (t + 1))
    return {}
cats = {}
for q in ['360° panoramas of restaurants', '360° panoramas of cafés', '360° panoramas of bars', '360° panoramas of pubs', 'spherical panoramas restaurants',
          '360° panoramas of interiors of restaurants', 'equirectangular restaurant', '360° panoramas of shops', '360° panoramas of food courts', '360 panoramas Korea interior', '360° panoramas of Japan interiors']:
    r = api(action='query', list='search', srnamespace=14, srsearch=q, srlimit=20)
    for x in r.get('query', {}).get('search', []): cats[x['title']] = 1
    time.sleep(.5)
info = []
for c in sorted(cats):
    r = api(action='query', prop='categoryinfo', titles=c); time.sleep(.3)
    ci = r.get('query', {}).get('pages', [{}])[0].get('categoryinfo', {})
    if re.search(r'360|spheric|equirect|panoram', c, re.I): info.append((ci.get('files', 0), ci.get('subcats', 0), c))
for f, s, c in sorted(info, reverse=True)[:50]: print(f'{f:6d} f {s:4d} s  {c}')
