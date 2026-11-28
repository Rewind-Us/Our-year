"""잇기 v1 팀원 테스트 페이지 (맥미니 전용).
사진 여러 장을 올리면 같은 공간 묶음, 사진 쌍 점수와 판정, 확인 질문 후보를 보여 준다.
평가 때와 같은 순서: 긴 변 1280 축소 -> macOS Vision 사람 마스크 -> 322x322 -> MegaLoc+LoRA v1 -> 같은 환경 점수 -> 라우터.
실행: ~/our-year-train/.venv/bin/python linking_demo.py [--share]
"""
import sys, json, subprocess, tempfile, os
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageOps
import pillow_heif; pillow_heif.register_heif_opener()
H = Path.home() / 'our-year-train'
sys.path.insert(0, str(H / 'vt'))
from common import megaloc, MEAN, STD, dev
import gradio as gr

APP = Path(__file__).resolve().parent
CFG = json.load(open(APP / 'config.json'))
SEG = str(H / 'pubeval/seg')
CKPT = str(H / 'vt/run_f1/step2000.ckpt')
G = 23
net = megaloc(4, 8, CKPT)
sys.path.insert(0, str(APP)); from router import route, groups


def chamfer_mat(A, B, ma, mb):
    S = torch.einsum('ipd,jqd->ijpq', A, B)
    S = S.masked_fill(mb[None, :, None, :], -2).masked_fill(ma[:, None, :, None], -2)
    va, vb = (~ma).float(), (~mb).float()
    ab = (S.max(3).values * va[:, None]).sum(2) / va.sum(1)[:, None].clamp(min=1)
    ba = (S.max(2).values.clamp(min=-1) * vb[None]).sum(2) / vb.sum(1)[None].clamp(min=1)
    return .5 * (ab + ba)


def prep(path, work):
    im = ImageOps.exif_transpose(Image.open(path)).convert('RGB'); im.thumbnail((1280, 1280), Image.LANCZOS)
    j = os.path.join(work, 'a.jpg'); m = os.path.join(work, 'a_m.png'); im.save(j, quality=92)
    subprocess.run([SEG, j, m], check=True, capture_output=True)
    x = np.asarray(im.resize((322, 322), Image.BICUBIC), dtype=np.float32) / 255
    mk = (np.asarray(Image.open(m).convert('L').resize((322, 322), Image.NEAREST)) > 127).astype(np.float32)
    pm = mk.reshape(G, 14, G, 14).mean((1, 3)) > .5
    return im, (torch.from_numpy(x).permute(2, 0, 1) - MEAN) / STD, torch.from_numpy(pm).flatten(), float(mk.mean())


def run_small(payload):
    """브라우저에서 줄인 사진(JSON: [{name, data(base64 jpeg)}])을 받아 판정한다."""
    import base64
    items = json.loads(payload or '[]')
    if len(items) < 2:
        return '<p>사진을 2장 이상 골라 주세요.</p>', None
    tmp = tempfile.mkdtemp(); paths = []
    for k, it in enumerate(items):
        p = os.path.join(tmp, f"{k:02d}_" + Path(it['name']).stem[:40] + '.jpg')
        open(p, 'wb').write(base64.b64decode(it['data'].split(',', 1)[1])); paths.append(p)
    try:
        return run(paths)
    finally:
        for p in paths: os.remove(p)
        os.rmdir(tmp)


import base64, io, html as _h
T_D, T_B = CFG['router']['direct_threshold'], CFG['router']['bridge_threshold']


def _thumb(im, h=170):
    t = im.copy(); t.thumbnail((h * 2, h)); buf = io.BytesIO(); t.save(buf, 'JPEG', quality=80)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()


def _card(k, names, ims, pf, h=170):
    note = ' (사람이 많이 가림)' if pf[k] >= .5 else ''
    return (f'<div style="display:inline-block;margin:4px;text-align:center;vertical-align:top">'
            f'<img src="{_thumb(ims[k], h)}" style="height:{h}px;border-radius:8px;border:1px solid #ccc"><br>'
            f'<span style="font-size:13px"><b>사진 {k + 1}</b> {_h.escape(names[k].split(". ", 1)[1])}{note}</span></div>')


def render(names, ims, S, res, grp, pf):
    N = len(names); multi = [g for g in grp if len(g) > 1]
    asks = [(i, j) for (i, j), v in res.items() if v == 'ask']
    in_ask = {x for p in asks for x in p}
    alone = [g[0] for g in grp if len(g) == 1 and g[0] not in in_ask]
    box = 'border:1px solid #ddd;border-radius:12px;padding:14px;margin:12px 0'
    o = [f'<div style="font-size:15px;line-height:1.6">',
         f'<div style="{box};background:#f7f7f7"><b>결과 요약</b>: 사진 {N}장 중 '
         f'<b>같은 공간 {len(multi)}곳</b>으로 자동 묶음, <b>같은 곳일 수도 있는 짝 {len(asks)}개</b>, '
         f'어디와도 이어지지 않은 사진 {len(alone)}장</div>']
    o.append('<h3>1. 같은 공간으로 묶은 사진 (자동)</h3>')
    if not multi:
        o.append('<p style="color:#666">자동으로 묶인 사진이 없어요.</p>')
    for gi, g in enumerate(multi):
        o.append(f'<div style="{box}"><b style="font-size:17px">공간 {gi + 1}</b> <span style="color:#666">사진 {len(g)}장</span><br>')
        o += [_card(k, names, ims, pf) for k in g]
        o.append('<div style="margin-top:8px;font-size:14px">어떻게 이어졌나요?<ul style="margin:4px 0">')
        for (i, j), v in sorted(res.items(), key=lambda kv: -S[kv[0]]):
            if i in g and j in g and v == 'direct':
                o.append(f'<li>사진 {i + 1} ↔ 사진 {j + 1}: 점수 {S[i, j]:.3f}, <b>바로 이어짐</b> (기준 {T_D} 넘음)</li>')
        for (i, j), v in sorted(res.items(), key=lambda kv: -S[kv[0]]):
            if i in g and j in g and v == 'bridge':
                ks = [k for k in range(N) if k not in (i, j) and S[i, k] > T_B and S[k, j] > T_B]
                via = ', '.join(f'사진 {k + 1}' for k in ks[:2])
                o.append(f'<li>사진 {i + 1} ↔ 사진 {j + 1}: 점수 {S[i, j]:.3f}로 직접은 부족하지만 <b>{via}을(를) 거쳐 이어짐</b></li>')
        o.append('</ul></div></div>')
    o.append('<h3>2. 같은 곳일 수도 있어요 (확인 필요)</h3>')
    if not asks:
        o.append('<p style="color:#666">없어요.</p>')
    else:
        o.append('<p style="color:#555;font-size:14px">서로 가장 닮은 짝이지만 점수가 기준보다 낮아 자동으로 잇지 않았어요. '
                 '실제 앱에서는 "같은 곳에서 찍었나요?"라고 물어봐요.</p>')
    for i, j in sorted(asks, key=lambda p: -S[p]):
        o.append(f'<div style="{box};border-style:dashed">')
        o.append(_card(i, names, ims, pf))
        o.append(f'<div style="display:inline-block;vertical-align:middle;margin:0 10px;text-align:center;font-size:14px">'
                 f'↔<br>점수 {S[i, j]:.3f}<br><span style="color:#666">(기준 {T_D})</span></div>')
        o.append(_card(j, names, ims, pf)); o.append('</div>')
    o.append('<h3>3. 어디와도 이어지지 않은 사진</h3>')
    o.append(''.join(_card(k, names, ims, pf, 120) for k in alone) or '<p style="color:#666">없어요.</p>')
    o.append(f'<div style="{box};background:#fafafa;font-size:14px"><b>읽는 법</b><ul style="margin:4px 0">'
             f'<li>점수는 두 사진의 배경이 얼마나 비슷한지예요. 사람 부분은 빼고 비교해요.</li>'
             f'<li>{T_D} 넘으면 바로 같은 공간, 직접은 부족해도 다른 사진 한 장과 양쪽 모두 {T_B} 넘으면 그 사진을 거쳐 같은 공간이에요.</li>'
             f'<li>같은 장소 사진을 <b>3장 이상</b> 넣으면 거쳐서 이어질 길이 많아져 더 잘 묶여요.</li>'
             f'<li>천장만 찍은 사진, 얼굴이 화면 대부분을 가린 셀카는 배경이 적어 잘 안 이어질 수 있어요.</li></ul></div></div>')
    return ''.join(o)


def run(files):
    try:
        return _run(files)
    except Exception as e:
        return f'<p>처리 중 문제가 생겼어요: {type(e).__name__}. 사진을 다시 골라 주세요.</p>', None


def _run(files):
    if not files or len(files) < 2:
        return '<p>사진을 2장 이상 올려 주세요.</p>', None
    names, ims, X, M, pf = [], [], [], [], []
    with tempfile.TemporaryDirectory() as work:
        for k, f in enumerate(files):
            p = f if isinstance(f, str) else f.name
            im, x, m, r = prep(p, work)
            names.append(f'{k + 1}. {Path(p).name}'); ims.append(im); X.append(x); M.append(m); pf.append(r)
    X = torch.stack(X).to(dev); M = torch.stack(M).to(dev)
    with torch.no_grad():
        pfeat, cls = net.m.backbone(X)
        P = torch.nn.functional.normalize(pfeat.flatten(2).transpose(1, 2), dim=2)
        S = torch.cat([chamfer_mat(P[i:i + 8], P, M[i:i + 8], M) for i in range(0, len(P), 8)]).cpu().numpy()
    res = route(S, CFG); grp = groups(res, len(names))
    grp.sort(key=len, reverse=True)
    lab = {'direct': '자동 연결(직접)', 'bridge': '자동 연결(다리)', 'ask': '확인 질문', 'no': '연결 안 함'}
    rows = [[names[i], names[j], round(float(S[i, j]), 3), lab[v]] for (i, j), v in sorted(res.items(), key=lambda kv: -S[kv[0]])]
    rows += [[names[i], '(사람 비율)', round(pf[i], 2), ''] for i in range(len(names))]
    return render(names, ims, S, res, grp, pf), rows


JS = "\n() => {\n  document.addEventListener('change', async (ev) => {\n    const inp = ev.target;\n    if (!inp || inp.id !== 'pick') return;\n    const st = document.getElementById('pickstate'); const out = [];\n    st.textContent = '사진 줄이는 중...';\n    for (const f of inp.files) {\n      try {\n        const bmp = await createImageBitmap(f, {imageOrientation: 'from-image'});\n        const r = Math.min(1, 1280 / Math.max(bmp.width, bmp.height));\n        const c = document.createElement('canvas'); c.width = Math.round(bmp.width * r); c.height = Math.round(bmp.height * r);\n        c.getContext('2d').drawImage(bmp, 0, 0, c.width, c.height);\n        const blob = await new Promise(res => c.toBlob(res, 'image/jpeg', 0.85));\n        out.push(new File([blob], f.name.replace(/\\.[^.]+$/, '') + '.jpg', {type: 'image/jpeg'}));\n      } catch (e) { st.textContent = f.name + ' 은(는) 이 브라우저에서 열 수 없어요. 아이폰 Safari나 JPG 사진으로 해 주세요.'; return; }\n    }\n    const target = document.querySelector('#small input[type=file]');\n    const dt = new DataTransfer(); out.forEach(x => dt.items.add(x));\n    target.files = dt.files; target.dispatchEvent(new Event('change', {bubbles: true}));\n    const kb = Math.round(out.reduce((a, o) => a + o.size, 0) / 1024);\n    st.textContent = out.length + '장을 줄여서 올리는 중 (' + kb + 'KB). 아래 칸에 사진 목록이 다 뜨면 판정하기를 눌러 주세요.';\n  });\n}\n"
with gr.Blocks(title='잇기 v1 테스트', delete_cache=(600, 3600)) as demo:
    gr.Markdown('# 잇기 v1: 같은 공간 묶기 테스트\n사진을 여러 장 고르고, 아래 칸에 목록이 다 뜨면 **판정하기**를 누르세요. 같은 장소 사진은 3장 이상 넣으면 더 잘 묶여요. 사진은 브라우저에서 줄인 뒤 보내요. 같은 공간에서 찍은 사진끼리 묶어 보여 줘요. 올린 사진은 맥미니 임시 폴더에 잠시 남았다가 1시간 안에 자동으로 지워져요.')
    gr.HTML('<label style="font-size:16px">사진 고르기 (여러 장): <input id="pick" type="file" accept="image/*" multiple></label><div id="pickstate" style="margin-top:6px;color:#555"></div>')
    small = gr.File(file_count='multiple', elem_id='small', label='줄인 사진 (자동으로 채워져요)', interactive=True)
    btn = gr.Button('판정하기', variant='primary')
    out = gr.HTML()
    with gr.Accordion('자세한 점수 표', open=False):
        tab = gr.Dataframe(headers=['사진 A', '사진 B', '점수', '판정'], label='사진 쌍 점수')
    btn.click(run, small, [out, tab])
    with gr.Accordion('원본 파일로 올리기 (느려요)', open=False):
        inp = gr.File(file_count='multiple', file_types=['image', '.heic', '.HEIC'], label='사진 올리기')
        btn2 = gr.Button('원본으로 판정하기')
        btn2.click(run, inp, [out, tab])

if __name__ == '__main__':
    auth = ('team', (APP / '.password').read_text().strip()) if '--auth' in sys.argv else None
    demo.queue(max_size=8).launch(server_name='127.0.0.1', server_port=7860, share='--share' in sys.argv, auth=auth, js=JS)
