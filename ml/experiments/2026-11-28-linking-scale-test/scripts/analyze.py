import numpy as np, itertools, json
z=np.load('stress.npz'); n=list(z['names']); g=list(z['groups']); S=z['S']; Gl=z['Gl']; N=len(n)
T,TB=.611,.631
same=np.array([[a==b for b in g] for a in g]); np.fill_diagonal(same,False)
def comps(E):
    par=list(range(N))
    def f(x):
        while par[x]!=x: par[x]=par[par[x]]; x=par[x]
        return x
    I,J=np.nonzero(np.triu(E,1))
    for i,j in zip(I,J): par[f(i)]=f(j)
    lab=np.array([f(i) for i in range(N)]); return lab
def evaluate(K, pools=None, tag=''):
    """K: 비교 허용 쌍(bool NxN). pools: 사진별 묶음 id (같은 pool 안에서만 비교)"""
    if pools is not None: K=K&(pools[:,None]==pools[None,:])
    np.fill_diagonal(K,False)
    D=K&(S>T); B=(K&(S>TB)).astype(np.int32)
    E=D|(K&((B@B)>0))
    lab=comps(E); L=lab[:,None]==lab[None,:]; np.fill_diagonal(L,False)
    out={}
    for kind,mask in (('real',np.array([not x.startswith('pano:') for x in g])),('pano',np.array([x.startswith('pano:') for x in g]))):
        places={}
        for i in np.nonzero(mask)[0]: places.setdefault(g[i],[]).append(i)
        multi=[ix for ix in places.values() if len(ix)>1]
        whole=sum(1 for ix in multi if len(set(lab[ix]))==1)
        sp=sum(L[i,j] for ix in multi for i,j in itertools.combinations(ix,2)); tp=sum(1 for ix in multi for _ in itertools.combinations(ix,2))
        out[kind]=(whole,len(multi),int(sp),tp)
    fp=int((L&~same).sum()//2)
    # 오염된 묶음 수
    bad=0
    for l in set(lab):
        ix=np.nonzero(lab==l)[0]
        if len({g[i] for i in ix})>1: bad+=1
    print(f"{tag:28s} 비교쌍 {int(K.sum()//2):6d} | 실사진 장소 온전 {out['real'][0]}/{out['real'][1]}, 같은장소쌍 {out['real'][2]}/{out['real'][3]} | 파노라마 장소 온전 {out['pano'][0]}/{out['pano'][1]} | 다른장소 섞인 쌍 {fp}, 섞인 묶음 {bad}")
    return lab
full=np.ones((N,N),bool)
print('== A. 시간으로 안 좁힘 (최악), 348장 한 묶음')
evaluate(full.copy(),tag='전체 비교')
M=Gl.copy(); np.fill_diagonal(M,-9)
for k in (5,10,15,20,30):
    top=np.argsort(-M,1)[:,:k]; K=np.zeros((N,N),bool)
    for i in range(N): K[i,top[i]]=True
    K|=K.T; evaluate(K,tag=f'전역 상위 {k}')

import collections
lab=evaluate(full.copy(),tag='(섞인 묶음 확인)')
for l in set(lab):
    ix=np.nonzero(lab==l)[0]; c=collections.Counter(g[i] for i in ix)
    if len(c)>1: print('  섞인 묶음', len(ix),'장:', dict(c.most_common(6)), '...' if len(c)>6 else '')

# ---- B. 시간 정보 부여 ----
import datetime as dt
known=json.load(open('times_private.json'))   # 실사진 촬영 시각: 개인 정보라 저장소에 넣지 않음
real_groups=sorted({x for x in g if not x.startswith('pano:')})
rng=np.random.default_rng(0)
base={}  # 장소별 기준 시각(분)
def mins(s): return dt.datetime.strptime(s,'%Y-%m-%d %H:%M:%S').timestamp()/60
for i,x in enumerate(n):
    if x in known: base.setdefault(g[i],mins(known[x]))
day0=mins('2026-09-01 10:00:00')
for k,rg_ in enumerate(real_groups):
    base.setdefault(rg_, day0+k*1440*3)   # 시각을 모르는 장소는 3일 간격으로
t=np.zeros(N)
for i in range(N):
    if not g[i].startswith('pano:'): t[i]=base[g[i]]+rng.uniform(0,15)
def run_time(mode):
    pg=sorted({x for x in g if x.startswith('pano:')}); rb=list({base[x] for x in real_groups})
    for x in pg:
        if mode=='random': b=mins('2026-06-15 00:00:00')+rng.uniform(0,170*1440)
        else: b=rb[rng.integers(len(rb))]+rng.uniform(-90,90)   # 실제 사건과 같은 시간대(최악)
        for i in range(N):
            if g[i]==x: t[i]=b+rng.uniform(0,15)
    # 사건 나누기: 시간순 정렬 후 간격 > GAP 이면 끊기
    for GAP in (60,120,240):
        o=np.argsort(t); ev=np.zeros(N,int); e=0
        for a,b in zip(o[:-1],o[1:]):
            ev[a]=e
            if t[b]-t[a]>GAP: e+=1
        ev[o[-1]]=e
        sizes=np.bincount(ev)
        M=Gl.copy(); np.fill_diagonal(M,-9)
        for k in (None,10):
            K=np.ones((N,N),bool)
            if k:
                # 상위 k는 같은 사건 안에서 고른다
                Mm=np.where(ev[:,None]==ev[None,:],M,-9)
                top=np.argsort(-Mm,1)[:,:k]; K=np.zeros((N,N),bool)
                for i in range(N): K[i,top[i]]=True
                K|=K.T
            evaluate(K,pools=ev,tag=f'{mode} 간격{GAP}분 사건{len(sizes)}개(최대{sizes.max()}장) '+(f'상위{k}' if k else '전체'))
print('== B1. 다른 장소들이 아무 날에나 찍힘')
run_time('random')
print('== B2. 다른 장소들이 실제 사건과 같은 시간대(±90분)에 찍힘 (최악)')
run_time('overlap')
