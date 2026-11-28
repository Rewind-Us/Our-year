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


def run(files):
    if not files or len(files) < 2:
        return '사진을 2장 이상 올려 주세요.', [], None
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
    lines = ['### 같은 공간 묶음']
    gal = []
    for gi, g in enumerate(grp):
        tag = f'공간 {gi + 1}' if len(g) > 1 else '혼자'
        lines.append(f'- **{tag}**: ' + ', '.join(names[i] for i in g))
        gal += [(ims[i], f'{tag} | {names[i]}') for i in g]
    ask = [(names[i], names[j], S[i, j]) for (i, j), v in res.items() if v == 'ask']
    lines.append('\n### 확인 질문 후보 ("같은 곳에서 찍었나요?")')
    lines += [f'- {a} 와 {b} (점수 {s:.3f})' for a, b, s in ask] or ['- 없음']
    r = CFG['router']
    lines.append(f"\n기준값: 직접 {r['direct_threshold']}, 다리 {r['bridge_threshold']}. 사람 비율이 높은 사진(60% 이상)은 점수가 불안정할 수 있어요.")
    lab = {'direct': '자동 연결(직접)', 'bridge': '자동 연결(다리)', 'ask': '확인 질문', 'no': '연결 안 함'}
    rows = [[names[i], names[j], round(float(S[i, j]), 3), lab[v]] for (i, j), v in sorted(res.items(), key=lambda kv: -S[kv[0]])]
    rows += [[names[i], '(사람 비율)', round(pf[i], 2), ''] for i in range(len(names))]
    return '\n'.join(lines), gal, rows


with gr.Blocks(title='잇기 v1 테스트', delete_cache=(600, 3600)) as demo:
    gr.Markdown('# 잇기 v1: 같은 공간 묶기 테스트\n사진을 여러 장(5장 안팎) 한 번에 올리고 **판정하기**를 누르세요. 같은 공간에서 찍은 사진끼리 묶어 보여 줘요. 올린 사진은 맥미니 임시 폴더에 잠시 남았다가 1시간 안에 자동으로 지워져요.')
    inp = gr.File(file_count='multiple', file_types=['image', '.heic', '.HEIC'], label='사진 올리기')
    btn = gr.Button('판정하기', variant='primary')
    out = gr.Markdown()
    gal = gr.Gallery(label='묶음별 사진', columns=5, height='auto')
    tab = gr.Dataframe(headers=['사진 A', '사진 B', '점수', '판정'], label='사진 쌍 점수')
    btn.click(run, inp, [out, gal, tab])

if __name__ == '__main__':
    auth = ('team', (APP / '.password').read_text().strip()) if '--auth' in sys.argv else None
    demo.queue(max_size=8).launch(server_name='127.0.0.1', server_port=7860, share='--share' in sys.argv, auth=auth)
