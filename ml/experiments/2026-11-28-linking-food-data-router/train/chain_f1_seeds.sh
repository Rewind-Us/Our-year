#!/bin/zsh
# run_f1과 같은 설정(식당, 카페, 야외 파노라마 99장 추가)을 시드 1, 2로 다시 학습한다.
# train_pano_f.py 는 ../2026-11-27-linking-env-score-lora/train/ 에 있다.
cd ~/our-year-train/vt; PY=../.venv/bin/python
st() { echo "$1 $2" >> chain_f1s.status; }
for s in 1 2; do
$PY train_pano_f.py --out run_f1s$s --steps 2000 --kd 0 --blocks 4 --group 1 --seed $s --food 1 --outdoor 1 --gc 1 > run_f1s$s.log 2>&1; st run_f1s$s $?
done
st ALLDONE 0
