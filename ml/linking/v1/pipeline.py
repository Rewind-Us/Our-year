"""잇기 v1 두 사람 사진 처리 흐름.

나와 친구의 사진(각자 수백 장)에서 같은 순간, 같은 공간에서 찍은 사진 묶음을 찾는다.

1. 각자의 사진을 촬영 시각으로 사건으로 나눈다(간격 > event_gap_min 이면 끊음).
2. 두 사람의 사건 중 시간이 겹치는 것(앞뒤 cross_tolerance_min 여유)을 하나의 비교 묶음으로 합친다.
3. 비교 묶음 안에서 사진마다 전역 벡터가 가장 비슷한 top_k 장만 고른다(나와 친구, 같은 사람끼리 모두).
4. 고른 쌍만 같은 환경 점수로 정밀 비교한다(score_fn).
5. 라우터 v1 규칙(직접, 다리)으로 잇고, 두 사람의 사진이 모두 들어간 묶음만 결과로 낸다.
   직접, 다리로 이어지지 않은 나와 친구 사이 쌍 중 상대 순위 규칙에 맞는 것은 확인 질문 후보로 낸다.

촬영 시각이 없는 사진은 비교하지 않고 따로 돌려준다(no_time).
점수 계산은 밖에서 넣어 준다. 서버든 기기든 같은 흐름을 쓸 수 있게 하기 위해서다.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np


def load_config(path=None):
    return json.load(open(path or Path(__file__).with_name('config.json')))


@dataclass
class Photo:
    id: str
    owner: str            # 'a' 또는 'b'
    time: Optional[float]  # 촬영 시각(분 단위 숫자). 없으면 None


@dataclass
class Result:
    moments: list = field(default_factory=list)   # [{'photos': [id...], 'pool': k}]
    ask: list = field(default_factory=list)       # [(id_a, id_b, score)]
    no_time: list = field(default_factory=list)   # 시각이 없어 비교하지 않은 사진 id
    stats: dict = field(default_factory=dict)


def split_events(times: np.ndarray, gap: float) -> np.ndarray:
    """시간순으로 정렬해 간격이 gap보다 크면 새 사건. 사진별 사건 번호를 돌려준다."""
    ev = np.zeros(len(times), int)
    if len(times) == 0:
        return ev
    order = np.argsort(times); e = 0
    for a, b in zip(order[:-1], order[1:]):
        ev[a] = e
        if times[b] - times[a] > gap:
            e += 1
    ev[order[-1]] = e
    return ev


def make_pools(photos: Sequence[Photo], gap: float, tol: float):
    """두 사람의 사건 중 시간이 겹치는 것끼리 합친 비교 묶음. 반환: {pool번호: [사진 index]}"""
    events = []  # (owner, start, end, [index])
    for owner in ('a', 'b'):
        ix = [i for i, p in enumerate(photos) if p.owner == owner and p.time is not None]
        if not ix:
            continue
        t = np.array([photos[i].time for i in ix]); ev = split_events(t, gap)
        for e in set(ev.tolist()):
            mem = [ix[k] for k in np.nonzero(ev == e)[0]]
            ts = [photos[i].time for i in mem]
            events.append((owner, min(ts), max(ts), mem))
    par = list(range(len(events)))
    def f(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for i, (oi, si, ei, _) in enumerate(events):
        for j, (oj, sj, ej, _) in enumerate(events):
            if oi == 'a' and oj == 'b' and si - tol <= ej and sj - tol <= ei:
                par[f(i)] = f(j)
    pools = {}
    for i, (o, s, e, mem) in enumerate(events):
        pools.setdefault(f(i), []).extend(mem)
    owners = {k: {photos[i].owner for i in v} for k, v in pools.items()}
    # 한 사람 사진만 있는 묶음은 같은 순간이 될 수 없으므로 뺀다
    return {k: v for k, v in pools.items() if owners[k] == {'a', 'b'}}


def link(photos: Sequence[Photo], glob: np.ndarray, score_fn: Callable[[list], np.ndarray], cfg=None) -> Result:
    """photos: 사진 목록, glob: 사진별 전역 벡터(정규화, N x D), score_fn: [(i, j)...] -> 같은 환경 점수 배열"""
    cfg = cfg or load_config(); r = cfg['router']; pc = cfg['pipeline']
    T, TB, MG, FLOOR = r['direct_threshold'], r['bridge_threshold'], r['ask']['min_gap'], r['ask']['min_score']
    res = Result(no_time=[p.id for p in photos if p.time is None])
    pools = make_pools(photos, pc['event_gap_min'], pc['cross_tolerance_min'])
    n_pairs = 0
    for pk, ix in pools.items():
        G = glob[ix] @ glob[ix].T; np.fill_diagonal(G, -9)
        k = min(pc['top_k'], len(ix) - 1)
        cand = set()
        for a in range(len(ix)):
            for b in np.argsort(-G[a])[:k]:
                cand.add((min(a, int(b)), max(a, int(b))))
        cand = sorted(cand); n_pairs += len(cand)
        sc = np.asarray(score_fn([(ix[a], ix[b]) for a, b in cand]), dtype=float)
        S = np.full((len(ix), len(ix)), -1.0)
        for (a, b), s in zip(cand, sc):
            S[a, b] = S[b, a] = s
        # 직접, 다리 (후보 쌍 안에서만)
        D = S > T; Bm = (S > TB).astype(np.int32); E = D | (((Bm @ Bm) > 0) & (S > -1))
        par = list(range(len(ix)))
        def f(x):
            while par[x] != x:
                par[x] = par[par[x]]; x = par[x]
            return x
        for a, b in zip(*np.nonzero(np.triu(E, 1))):
            par[f(a)] = f(b)
        groups = {}
        for a in range(len(ix)):
            groups.setdefault(f(a), []).append(a)
        for mem in groups.values():
            if {photos[ix[a]].owner for a in mem} == {'a', 'b'}:
                res.moments.append({'photos': [photos[ix[a]].id for a in mem], 'pool': int(pk)})
        # 확인 질문: 나와 친구 사이, 자동으로 안 이어진 쌍
        lab = np.array([f(a) for a in range(len(ix))])
        for a, b in cand:
            if photos[ix[a]].owner == photos[ix[b]].owner or lab[a] == lab[b]:
                continue
            oa = np.sort(S[a][S[a] > -1])[::-1]; ob = np.sort(S[b][S[b] > -1])[::-1]
            if S[a, b] < oa[0] or S[a, b] < ob[0]:
                continue
            gap = min(oa[0] - (oa[1] if len(oa) > 1 else 0), ob[0] - (ob[1] if len(ob) > 1 else 0))
            if gap >= MG and S[a, b] >= FLOOR:
                res.ask.append((photos[ix[a]].id, photos[ix[b]].id, float(S[a, b])))
    res.stats = {'photos': len(photos), 'pools': len(pools), 'pairs_scored': n_pairs,
                 'pairs_all_cross': sum(p.owner == 'a' for p in photos) * sum(p.owner == 'b' for p in photos)}
    return res
