import sys, random, math, torch, torch.nn.functional as F
sys.argv_M=sys.argv[1] if len(sys.argv)>1 else "64"
sys.argv=["x"]; import arch_vet_p49 as Q
Q.PKB=True; Q.ROLE=4; Q.BMODE="oracle"; Q.SURPRISE=False
m=Q.CKB(256,32,64); m.load_state_dict(torch.load("p21_ckpt/c103rbk_CKB_d32_s1.pt")); m.eval()
x,spans=Q.dense_batch(random.Random(996),8,Q.NAMES_OOD,6,320)
B,L=x.shape; d=32; M=int(sys.argv_M) if hasattr(sys,"argv_M") else 64; tau=0.05
with torch.no_grad():
    e=m.E(x); h=torch.zeros(B,d); c=torch.zeros(B,d); dk=d+4
    K0=torch.cat([m.K0, torch.randn(max(M-64,0),32)*0.5],0)[:M]; K=torch.cat([K0,torch.zeros(M,4)],-1).unsqueeze(0).expand(B,M,dk).clone(); Vm=torch.zeros(B,M,d); occ=torch.zeros(B,M); pk=torch.zeros(B,dk)
    hits=tot=0; evict=0; present=0; conf_sum=0; mind=[]
    namepos={(i,p) for i,sp in enumerate(spans) for s0,s1 in sp for p in range(s0,s1-1)}  # predicting byte p+1 from position p
    for t in range(L-1):
        h=m.gh(e[:,t],h); c=m.gc(e[:,t],c); xt=x[:,t]
        b=((xt==32)|(xt==10)|(xt==44)|(xt==46)|(xt==58)|(xt==63)|(xt==33)).float().unsqueeze(-1)
        logocc=torch.log(occ+1e-4); qv=torch.cat([c,torch.tanh(m.z(h))],-1)
        sim=torch.einsum("bd,bmd->bm",qv,K)/math.sqrt(d); a=torch.softmax((sim+logocc)/tau,-1); r=torch.einsum("bm,bmd->bd",a,Vm)
        # read accuracy at name positions: nearest embedding to r vs true next byte
        for i in range(B):
            if (i,t) in namepos:
                tot+=1; tgt=x[i,t+1].item()
                nn_=torch.cdist(r[i:i+1],m.E.weight).argmin().item(); hits+=int(nn_==tgt); conf_sum+=a[i].max().item()
                # is any slot holding a key within small distance of qv with value == e[tgt]? (binding present)
                dv=torch.cdist(Vm[i],m.E.weight[tgt:tgt+1]).squeeze(-1); dkq=torch.cdist(K[i],qv[i:i+1]).squeeze(-1)
                j=dv.argmin().item(); mind.append((round(dv[j].item(),2), round(dkq[j].item(),2), round(occ[i,j].item(),2))); present+=int(dv[j].item()<0.5)
        wsim=torch.einsum("bd,bmd->bm",pk,K)/math.sqrt(d); mm=torch.softmax((wsim+logocc)/tau,-1)
        p=torch.sigmoid(wsim.max(-1,keepdim=True).values-m.theta); empty=torch.softmax(-occ/tau,-1)
        alpha=(p*mm+(1-p)*empty).unsqueeze(-1)
        K=(1-alpha)*K+alpha*pk.unsqueeze(1); Vm=(1-alpha)*Vm+alpha*e[:,t].unsqueeze(1); occ=occ+alpha.squeeze(-1)*(1-occ)
        pk=qv; c=c*(1-b)
import statistics; print("M",M,"best-value slot: median (value-dist, key-dist, occ)", [round(statistics.median([u[k] for u in mind]),2) for k in range(3)], "emb norm", round(m.E.weight.norm(dim=1).mean().item(),2)); print(f"name-byte positions {tot}: read-nearest==target {hits/tot:.3f}; binding present in memory {present/tot:.3f}; mean max-attn {conf_sum/tot:.2f}; theta {m.theta.item():.2f}")
