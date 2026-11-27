import json, numpy as np
from PIL import Image, ImageDraw, ImageFont
R = json.load(open('match.json')); S = 420; G = 23; c = S / G
FONT = '/System/Library/Fonts/AppleSDGothicNeo.ttc'; f1 = ImageFont.truetype(FONT, 22); f2 = ImageFont.truetype(FONT, 17)
env = json.loads(open('res.txt').read().strip().splitlines()[-1]); names = env['names']
def heat(v):  # 0.3 이하 파랑 -> 0.9 이상 빨강
    t = np.clip((v - .3) / .6, 0, 1); return (int(255 * t), int(80 + 60 * (1 - abs(t - .5) * 2)), int(255 * (1 - t)))
def panel(tag, a, b):
    d = R[f'{tag}|{a}-{b}']; v, k, ma = np.array(d['v']), np.array(d['k']), np.array(d['ma'])
    A = Image.open(f'up/i_{a}.png').convert('RGB').resize((S, S)); B = Image.open(f'up/i_{b}.png').convert('RGB').resize((S, S))
    mb = (np.asarray(Image.open(f'up/m_{b}.png').convert('L').resize((G, G), Image.BOX)) > 127).flatten()
    ov = Image.new('RGBA', (S, S), (0, 0, 0, 0)); o = ImageDraw.Draw(ov)
    for p in range(G * G):
        x, y = p % G * c, p // G * c
        o.rectangle([x, y, x + c, y + c], fill=(40, 40, 40, 190) if ma[p] else heat(v[p]) + (120,))
    A = Image.alpha_composite(A.convert('RGBA'), ov).convert('RGB')
    ovb = Image.new('RGBA', (S, S), (0, 0, 0, 0)); ob = ImageDraw.Draw(ovb)
    for p in np.where(mb)[0]: x, y = p % G * c, p // G * c; ob.rectangle([x, y, x + c, y + c], fill=(40, 40, 40, 190))
    B = Image.alpha_composite(B.convert('RGBA'), ovb).convert('RGB')
    W = Image.new('RGB', (2 * S + 40, S + 50), 'white'); W.paste(A, (0, 50)); W.paste(B, (S + 40, 50)); dr = ImageDraw.Draw(W)
    i, j = names.index(a), names.index(b); sc = env[f'{tag}_env'][i][j]; thr = {'base': .614, 'g0': .629}[tag]
    dr.text((0, 10), f"{'원래 모델' if tag == 'base' else '학습 모델 run_g0'}   점수 {sc:.3f} (기준선 {thr}) {'통과' if sc > thr else '미통과'}", font=f1, fill='black')
    # 상위 매칭 20개 연결선(서로 떨어진 조각만)
    order = [p for p in np.argsort(-v) if not ma[p]]; sel = []
    for p in order:
        if all(abs(p % G - q % G) + abs(p // G - q // G) >= 3 for q in sel): sel.append(p)
        if len(sel) == 20: break
    for p in sel:
        x0, y0 = p % G * c + c / 2, 50 + p // G * c + c / 2; q = k[p]; x1, y1 = S + 40 + q % G * c + c / 2, 50 + q // G * c + c / 2
        dr.line([x0, y0, x1, y1], fill=(255, 230, 0), width=2); dr.ellipse([x0 - 4, y0 - 4, x0 + 4, y0 + 4], fill=(255, 230, 0)); dr.ellipse([x1 - 4, y1 - 4, x1 + 4, y1 + 4], fill=(255, 230, 0))
    return W
pairs = [('1690', '1693', '가장 높은 쌍'), ('1689', '1691', '높은 쌍'), ('1691', '1693', '가장 낮은 쌍'), ('1690', '1694', '낮은 쌍'), ('1689', 'q01_A', '다른 장소(대조군)')]
for a, b, t in pairs:
    p1, p2 = panel('base', a, b), panel('g0', a, b); H = Image.new('RGB', (p1.width, 60 + p1.height * 2 + 70), 'white'); d = ImageDraw.Draw(H)
    d.text((0, 12), f'IMG_{a} (왼쪽) vs {b if "q" in b else "IMG_" + b} (오른쪽) - {t}', font=ImageFont.truetype(FONT, 26), fill='black')
    H.paste(p1, (0, 60)); H.paste(p2, (0, 60 + p1.height))
    d.text((0, H.height - 62), '왼쪽 색: 각 조각이 오른쪽 사진에서 찾은 가장 닮은 조각과의 유사도 (파랑 낮음 -> 빨강 높음)', font=f2, fill='black')
    d.text((0, H.height - 36), '회색: 사람으로 판정돼 비교에서 뺀 조각   노란 선: 유사도 상위 매칭 20개', font=f2, fill='black')
    H.save(f'viz_{a}_{b}.png'); print(f'viz_{a}_{b}.png')
