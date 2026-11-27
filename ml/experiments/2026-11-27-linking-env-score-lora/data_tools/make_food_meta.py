"""pano_food: 제외 목록과 같은 장소 그룹, 하늘 비율을 만들어요."""
import json, re, numpy as np
from pathlib import Path
from PIL import Image
O = Path.home() / 'our-year-train/pubeval/raw/pano_food'; fs = sorted((O / 'img').glob('*.jpg'))
drop_idx = {18, 39, 43, 78, 83, 92, 93, 94, 95}
keep = [f.stem for k, f in enumerate(fs) if k not in drop_idx]
rules = [('VillaMedia_Wuppertal_Der_Gro', 'villamedia_saal'), ('VillaMedia_Wuppertal_Der_Wintergarten', 'villamedia_winter'), ('VillaMedia_Wuppertal_die_LichtBar', 'villamedia_lichtbar'),
         ('Uni_landau_cafeteria', 'landau_cafeteria'), ('Uni_landau_mensa', 'landau_mensa'), ('Wilhelm_D', 'wdg_mensa'), ('Darmstadtium', 'darmstadtium'),
         ('Basmanny_District', 'basmanny'), ('Brazil_santos_coffee_museum', 'santos_museum'), ('ALMA_OSF_canteen', 'alma_canteen'), ('Bad_D', 'bad_duerrenberg'),
         ('Leipzig_Kurt_Masur_Platz', 'leipzig_kmp'), ('Leipzig_Moritzbastei', 'leipzig_kmp'), ('Leipzig_Sch', 'leipzig_schoenefeld'), ('Cmglee_Wikimania2016_Esino_Lario', 'esino_lario')]
grp = {}
for s in keep:
    g = next((v for k, v in rules if s[3:].startswith(k)), s); grp[s] = 'food:' + g
sky = {}
for s in keep:
    a = np.asarray(Image.open(O / 'img' / f'{s}.jpg').convert('RGB').resize((512, 256))).astype(int)[:64]
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    sky[s] = round(float((((b > r + 12) & (b > 110)) | ((r > 225) & (g > 225) & (b > 225))).mean()), 3)
json.dump(dict(keep=keep, groups=grp, sky=sky), open(O / 'food_meta.json', 'w'), indent=0)
import collections; c = collections.Counter(grp.values())
print('dropped', [fs[k].stem[3:40] for k in sorted(drop_idx)]); print('keep', len(keep), 'groups', len(c), 'outdoor(sky>.2)', sum(v > .2 for v in sky.values()))
