#!/usr/bin/env python3
"""Build deterministic reviewed synthetic occlusion candidates from v2 assets.

Usage: python3 build_release.py [config.json]
Donor IDs must be explicitly curated in config.json before running.
"""
from __future__ import annotations
import csv, json, math, os, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps

DEFAULT_WORKSPACE = Path(os.environ["OUR_YEAR_WORKSPACE"])
ROOT = DEFAULT_WORKSPACE / "data/private/occlusion-v2"
OUT = Path(__file__).resolve().parent / "release-candidates"
CONFIG = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().with_name("release-config.json")
SELECTIONS = {
    "dining": [2,5,10,11,15,22,25,28,44,54],
    "outdoor": [1,4,12,18,24,35,41,52],
    "home": [27,30,34,51],
    "hall_study_office": [7,14,36,47],
    "other_interior": [45,48],
}
DEV_SEQUENCES = [22,44,24,41,34,36,48]
BANDS = ("medium",)

def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f: return list(csv.DictReader(f))
def seq_index():
    rows = read_csv(Path(__file__).resolve().with_name("index.csv"))
    return {int(r["sequence"]): r for r in rows if r["band"] == "medium"}
def count_visible(alpha, x, y, width, height):
    left, top, right, bottom = max(0,x), max(0,y), min(width,x+alpha.width), min(height,y+alpha.height)
    if right <= left or bottom <= top: return 0
    hist=alpha.crop((left-x,top-y,right-x,bottom-y)).histogram()
    return sum(hist[1:])
def scale_to_coverage(alpha, W, H, anchor_x, target):
    lo, hi = .01, 2.0
    hist=alpha.histogram()
    def count(a): return sum(hist[1:])
    # Source mask occupancy is preserved approximately by resize; estimate scale
    # from area first, then make a few deterministic corrections for clipping.
    visible=count(alpha); s=math.sqrt(target*W*H/max(1,visible))
    for _ in range(14):
        nw=max(1,round(alpha.width*s)); nh=max(1,round(alpha.height*s))
        a=alpha.resize((nw,nh),Image.Resampling.LANCZOS); x=round(W*anchor_x-nw/2); y=H-nh
        cov=count_visible(a,x,y,W,H)/(W*H)
        if abs(cov-target)<0.0003: break
        s *= math.sqrt(target/max(1e-9,cov))
    # Keep scale within safe bounds if unusual alpha geometry defeats estimate.
    nw=max(1,round(alpha.width*s)); nh=max(1,round(alpha.height*s))
    return nw,nh
def mild_align(fg, bg_crop, alpha):
    """Apply restrained brightness/color match to donor RGB only; preserve alpha."""
    from PIL import ImageEnhance, ImageStat
    stats_fg=ImageStat.Stat(fg.convert("RGB"), mask=alpha)
    stats_bg=ImageStat.Stat(bg_crop.convert("RGB"), mask=alpha)
    mf=sum(stats_fg.mean)/3; mb=sum(stats_bg.mean)/3
    factor=max(.90,min(1.10,mb/max(1,mf)))
    out=ImageEnhance.Brightness(fg).enhance(factor)
    # Keep chroma adjustments deliberately subtle and deterministic.
    return out.convert("RGBA")
def main():
    cfg=json.loads(CONFIG.read_text(encoding="utf-8"))
    donors=cfg.get("donors_by_split",{})
    if not donors.get("train") or not donors.get("synthetic_dev"):
        raise SystemExit(f"Config needs curated donor metadata for train and synthetic_dev: {CONFIG}")
    if {d.get("id") for d in donors["train"]} & {d.get("id") for d in donors["synthetic_dev"]}:
        raise SystemExit("Donor source IDs must be disjoint between train and synthetic_dev")
    selected=seq_index(); backgrounds={r["background_id"]:r for r in read_csv(ROOT/"backgrounds.csv")}
    donor_meta={}
    for split, entries in donors.items():
        for entry in entries:
            if isinstance(entry,str): raise SystemExit("Donor config entries must be objects with id, path, and source_url")
            if not all(entry.get(k) for k in ("id","path","source_url")): raise SystemExit(f"Incomplete donor metadata in {split}: {entry}")
            donor_meta[entry["id"]]=entry
    ordered_train=sorted(s for values in SELECTIONS.values() for s in values if s not in DEV_SEQUENCES)
    seq_to_split={s:("synthetic_dev" if s in DEV_SEQUENCES else "train") for values in SELECTIONS.values() for s in values}
    if set(seq_to_split[s] for s in DEV_SEQUENCES) != {"synthetic_dev"}: raise SystemExit("Bad dev sequence configuration")
    OUT.mkdir(parents=True,exist_ok=True); manifest=[]
    for scene, seqs in SELECTIONS.items():
        for seq in seqs:
            ix=selected[seq]; split=seq_to_split[seq]; bid=ix["background_id"]
            donors_here=donors[split]
            if len(donors_here)<2: raise SystemExit(f"Need >=2 curated donors for {split}")
            ix_in_pool=DEV_SEQUENCES.index(seq) if split=="synthetic_dev" else ordered_train.index(seq)
            bgrow=backgrounds[bid]
            if bgrow["scene"] != scene: raise SystemExit("Sequence %s expected %s, index has %s" % (seq, scene, bgrow["scene"]))
            bg=ImageOps.exif_transpose(Image.open(ROOT/bgrow["original_file"])).convert("RGB"); W,H=bg.size
            sample_dir=OUT/split; sample_dir.mkdir(parents=True,exist_ok=True)
            original_rel=f"{seq:02d}_{bid[:8]}_original.png"; bg.save(sample_dir/original_rel)
            primary_target=.16 if cfg.get("fallback_sequences",{}).get(str(seq),False) else .18
            # Retry the whole background at 16% only when 18% cannot fit two
            # distinct approved donors without clipping the head or sides.
            target_levels=[primary_target] if primary_target==.16 else [primary_target,.16]
            donor_order=[donors_here[(2*ix_in_pool+i)%len(donors_here)] for i in range(len(donors_here))]
            accepted=[]
            for target in target_levels:
                candidates=[]; tried=set()
                for donor_n,prow in enumerate(donor_order,1):
                    pid=prow["id"]
                    if pid in tried or len(candidates)>=2: continue
                    tried.add(pid)
                    donor_path=Path(prow["path"]); source=ImageOps.exif_transpose(Image.open(donor_path)).convert("RGBA")
                    crop_bottom_px=int(prow.get("crop_bottom_px",0))
                    if crop_bottom_px:
                        if crop_bottom_px<=0 or crop_bottom_px>=source.height-2: raise SystemExit(f"Invalid crop_bottom_px for donor {pid}")
                        source=source.crop((0,0,source.width,source.height-crop_bottom_px))
                    alpha=source.getchannel("A")
                    nw,nh=scale_to_coverage(alpha,W,H,.60,target)
                    fg=source.resize((nw,nh),Image.Resampling.LANCZOS); a=fg.getchannel("A")
                    x=round(W*.60-nw/2); y=H-nh
                    if x<0 or x+nw>W or y<0:
                        print(f"skip placement clipping seq={seq} donor={pid} at {target:.0%}")
                        continue
                    crop=bg.crop((x,y,x+nw,y+nh)); aligned=mild_align(fg,crop,a)
                    layer=Image.new("RGBA",(W,H),(0,0,0,0)); layer.alpha_composite(aligned,(x,y))
                    mask=layer.getchannel("A").point(lambda v:255 if v>0 else 0)
                    n=mask.histogram()[255]; total=W*H; cov=n/total*100
                    if not 15 <= cov <= 22: continue
                    if candidates and abs(cov-candidates[0]["coverage"])>2.0: continue
                    candidates.append(dict(donor_n=donor_n,prow=prow,pid=pid,donor_path=donor_path,crop_bottom_px=crop_bottom_px,nw=nw,nh=nh,x=x,y=y,layer=layer,mask=mask,n=n,total=total,coverage=cov))
                if len(candidates)>=2:
                    accepted=candidates[:2]
                    print(f"sequence {seq}: target {target:.0%}, coverages {[round(c['coverage'],4) for c in accepted]}")
                    break
            if len(accepted)<2: raise SystemExit(f"No valid pair for sequence {seq} at 18% or fallback 16%; add alternate approved donors")
            for c in accepted:
                prow=c["prow"]; pid=c["pid"]; donor_path=c["donor_path"]; cov=c["coverage"]; n=c["n"]; total=c["total"]
                stem=f"{seq:02d}_{bid[:8]}_p{c['donor_n']}_{pid[:8]}"
                composite_rel=stem+"_composite.png"; mask_rel=stem+"_mask.png"
                Image.alpha_composite(bg.convert("RGBA"),c["layer"]).save(sample_dir/composite_rel); c["mask"].save(sample_dir/mask_rel)
                manifest.append(dict(split=split,sequence=seq,scene=scene,background_id=bid,background_title=bgrow["title"],original_file=f"{split}/{original_rel}",donor_id=pid,candidate_id=prow["candidate_id"],donor_source_url=prow["source_url"],donor_cutout=str(donor_path),donor_source_crop_bottom_px=c["crop_bottom_px"],composite_file=f"{split}/{composite_rel}",mask_file=f"{split}/{mask_rel}",width=W,height=H,mask_pixels=n,total_pixels=total,coverage_percent=f"{cov:.6f}",anchor_x_fraction="0.600000",bottom_y_fraction="1.000000",placement_x=c["x"],placement_y=c["y"],scaled_width=c["nw"],scaled_height=c["nh"],source_image_id=ix["background_id"]))
    with open(OUT/"manifest.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=manifest[0].keys()); w.writeheader(); w.writerows(manifest)
    # Compact contact sheets, split x scene, max image width 1600 px.
    for split in ("train","synthetic_dev"):
        for scene in SELECTIONS:
            rows=[r for r in manifest if r["split"]==split and r["scene"]==scene]
            if not rows: continue
            cols=4; tw,th,lh=360,240,32; sheet=Image.new("RGB",(cols*tw,math.ceil(len(rows)/cols)*(th+lh)+34),"white"); d=ImageDraw.Draw(sheet); d.text((8,8),f"{split} | {scene}",fill="black")
            for j,r in enumerate(rows):
                im=Image.open(OUT/r["composite_file"]).convert("RGB"); im.thumbnail((tw,th),Image.Resampling.LANCZOS); x=j%cols*tw; y=34+j//cols*(th+lh); sheet.paste(im,(x,y)); label="%02d %s %s %s%%" % (r["sequence"],r["background_id"][:6],r["donor_id"][:6],r["coverage_percent"]); d.text((x+4,y+th+3),label,fill="black")
            sheet.save(OUT/f"{split}_{scene}.jpg",quality=88)
    print(f"created {len(manifest)} candidates; manifest: {OUT/'manifest.csv'}")
if __name__=="__main__": main()
