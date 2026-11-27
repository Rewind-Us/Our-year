import json, time, re
exec(open('search.py').read().split('cats = {}')[0])
Q = ['restaurant 360 panorama', 'restaurant equirectangular', 'restaurant spherical panorama', 'cafe 360 panorama', 'café 360° panorama', 'coffee shop 360 panorama',
     'bar interior 360 panorama', 'pub 360 panorama', 'Kneipe 360', 'Gaststätte 360 Panorama', 'Gasthaus 360 Panorama', 'bistro 360 panorama', 'izakaya 360',
     'canteen 360 panorama', 'Mensa 360 Panorama', 'cafeteria 360 panorama', 'food court 360 panorama', 'nightclub 360 panorama', 'brewery 360 panorama interior',
     'Brauhaus 360 Panorama', 'tavern 360 panorama', 'diner 360 panorama', 'tea house 360 panorama', 'bakery 360 panorama interior', 'Restaurant innen 360']
found = {}
for q in Q:
    off = 0
    while off < 200:
        r = api(action='query', list='search', srnamespace=6, srsearch=q, srlimit=50, sroffset=off); time.sleep(.4)
        hits = r.get('query', {}).get('search', [])
        for x in hits: found.setdefault(x['title'], q)
        if 'continue' not in r: break
        off = r['continue']['sroffset']
print('candidates', len(found))
FREE = re.compile(r'^(cc0|cc[ -]by(-sa)?[ -]?[\d.]*|public domain|pd.*|cc-by.*|cc-by-sa.*)', re.I)
ok = []; T = sorted(found)
for i in range(0, len(T), 40):
    r = api(action='query', titles='|'.join(T[i:i + 40]), prop='imageinfo', iiprop='url|size|extmetadata'); time.sleep(.6)
    for p in r.get('query', {}).get('pages', []):
        ii = (p.get('imageinfo') or [{}])[0]; w, h = ii.get('width', 0), (ii.get('height') or 1)
        lic = ii.get('extmetadata', {}).get('LicenseShortName', {}).get('value', '').strip()
        if 1.95 < w / h < 2.05 and w >= 3000 and FREE.match(lic) and p['title'].lower().endswith(('.jpg', '.jpeg')):
            ok.append(dict(title=p['title'], w=w, h=h, license=lic, q=found[p['title']]))
print('2:1 free', len(ok))
json.dump(ok, open('cand.json', 'w'), ensure_ascii=False, indent=0)
for o in ok: print(o['w'], o['license'][:12], o['title'][5:90])
