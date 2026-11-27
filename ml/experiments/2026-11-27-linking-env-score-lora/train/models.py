"""공개 평가용 장소 모델 4종을 같은 입력 규칙으로 불러와요. SALAD 집계층 코드는 serizba/salad(GPL-3.0)에서 가져온 내부 평가용이에요."""
import sys, torch, torch.nn as nn, torch.nn.functional as F, torchvision
from pathlib import Path
from safetensors.torch import load_file
ROOT = Path.home() / 'our-year-train'; PE = ROOT / 'pubeval'
HUB = Path.home() / '.cache/torch/hub/facebookresearch_dinov2_main'
MEAN = torch.tensor([.485, .456, .406]).view(3, 1, 1); STD = torch.tensor([.229, .224, .225]).view(3, 1, 1)
exec(open(PE / 'salad_agg.py').read())

def dino(pretrained=True):
    return torch.hub.load(str(HUB), 'dinov2_vitb14', source='local', pretrained=pretrained)

class Salad(nn.Module):
    def __init__(s):
        super().__init__(); s.bb = dino(False); s.agg = SALAD(num_channels=768, num_clusters=64, cluster_dim=128, token_dim=256)
        sd = torch.load(PE / 'w/dino_salad.ckpt', map_location='cpu'); sd = sd.get('state_dict', sd)
        new = {}
        for k, v in sd.items():
            if k.startswith('backbone.model.'): new['bb.' + k[15:]] = v
            elif k.startswith('aggregator.'): new['agg.' + k[11:]] = v
        miss = s.load_state_dict(new, strict=False); assert not miss.unexpected_keys and not [m for m in miss.missing_keys if 'mask_token' not in m], miss
    def forward(s, x):
        B, _, H, W = x.shape; t = s.bb.prepare_tokens_with_masks(x)
        for b in s.bb.blocks: t = b(t)
        t = s.bb.norm(t); f = t[:, 1:].reshape(B, H // 14, W // 14, 768).permute(0, 3, 1, 2)
        return s.agg((f, t[:, 0]))

class GeM(nn.Module):
    def forward(s, x, p=3, eps=1e-6): return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

class Eigen(nn.Module):
    def __init__(s):
        super().__init__(); r = torchvision.models.resnet50(); s.backbone = nn.Sequential(*list(r.children())[:-2])
        s.fc = nn.Linear(2048, 2048); sd = torch.load(PE / 'w/eigen_r50_2048.pth', map_location='cpu')
        bb = {k[9:]: v for k, v in sd.items() if k.startswith('backbone.')}; s.backbone.load_state_dict(bb)
        s.fc.load_state_dict({'weight': sd['aggregation.3.weight'], 'bias': sd['aggregation.3.bias']})
    def forward(s, x):
        x = F.normalize(s.backbone(x), dim=1); x = GeM()(x).flatten(1); return F.normalize(s.fc(x), dim=1)

class DinoMean(nn.Module):
    def __init__(s): super().__init__(); s.m = dino(True)
    def forward(s, x): return F.normalize(s.m.forward_features(x)['x_norm_patchtokens'].mean(1), dim=1)

class Mega(nn.Module):
    def __init__(s):
        super().__init__(); sys.path.insert(0, str(ROOT / 'model')); from megaloc_model import MegaLoc
        s.m = MegaLoc(); s.m.load_state_dict(load_file(ROOT / 'model/model.safetensors'))
    def forward(s, x): return F.normalize(s.m(x), dim=1)

SPECS = {'megaloc': (Mega, 322), 'salad': (Salad, 322), 'eigenplaces': (Eigen, 512), 'dinov2': (DinoMean, 322)}
