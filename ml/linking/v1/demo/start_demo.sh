#!/bin/zsh
# 테스트 페이지 + Cloudflare 임시 터널 시작. 링크는 tunnel.log에서 확인.
cd ~/our-year-train/app
pkill -f linking_demo.py; pkill -f "cloudflared tunnel"; sleep 2
: > demo.log; (PYTHONUNBUFFERED=1 nohup ../.venv/bin/python linking_demo.py > demo.log 2>&1 &)
until curl -s -o /dev/null http://127.0.0.1:7860/; do sleep 3; done
: > tunnel.log; (nohup /opt/homebrew/bin/cloudflared tunnel --no-autoupdate --url http://127.0.0.1:7860 > tunnel.log 2>&1 &)
until grep -q trycloudflare.com tunnel.log; do sleep 2; done
grep -Eo "https://[a-z0-9-]+\.trycloudflare\.com" tunnel.log | head -1
