#!/bin/zsh
# 휴대폰 사진 변형 실험: run_f1과 같은 설정에 --aug 1, 2, 3
cd ~/our-year-train/vt; PY=../.venv/bin/python
st() { echo "$1 $2 $(date +%H:%M)" >> chain_a.status; }
$PY train_pano_a.py --out run_asmoke --steps 4 --kd 0 --blocks 4 --group 1 --food 1 --outdoor 1 --aug 3 --eval_every 1000 --gc 1 > run_asmoke.log 2>&1; r=$?; st run_asmoke $r; [ $r -ne 0 ] && { st ALLDONE 1; exit 1; }
for A in 1 2 3; do
$PY train_pano_a.py --out run_a$A --steps 2000 --kd 0 --blocks 4 --group 1 --seed 0 --food 1 --outdoor 1 --aug $A --gc 1 > run_a$A.log 2>&1; st run_a$A $?
done
st ALLDONE 0
