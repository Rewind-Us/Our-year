#!/bin/zsh
# a1~a3 학습이 끝날 때마다 실사진 60장 점수 계산
cd ~/rtmp8; P=~/our-year-train/.venv/bin/python; V=~/our-year-train/vt
for A in 1 2 3; do
  until grep -q "^run_a$A " $V/chain_a.status 2>/dev/null; do sleep 60; done
  $P test6_a$A.py all60 > all60_a$A.json 2> all60_a$A.err; echo "a$A $? $(date +%H:%M)" >> eval_a.status
done
echo "ALLDONE $(date +%H:%M)" >> eval_a.status
