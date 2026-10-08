import sys, random, math, torch, torch.nn.functional as F
sys.argv_M=sys.argv[1] if len(sys.argv)>1 else "64"
ARGV=list(sys.argv); sys.argv=["x"]; import arch_vet_p49 as Q
Q.PKB=True; Q.ROLE=4; Q.BMODE="oracle"; Q.SURPRISE=False
PURE=len(ARGV)>2; Q.PUREADDR=({"lru":2,"hard3":3}.get(ARGV[2],1)) if PURE else 0; CK={"pure":"c103b","lru":"c103c","hard3":"c103d"}.get(ARGV[2] if PURE else "","c103rbk"); Q.ROLE=4; m=Q.CKB(256,32,256 if PURE else 64); m.load_state_dict(torch.load(f"p21_ckpt/{CK}_CKB_d32_s1.pt")); m.eval()
x,spans=Q.dense_batch(random.Random(996),8,Q.NAMES_OOD,6,320)
B,L=x.shape; d=32; M=int(sys.argv_M) if hasattr(sys,"argv_M") else 64; tau=0.05
with torch.no_grad():
    e=m.E(x); h=torch.zeros(B,d); c=torch.zeros(B,d); dk=d+4
    K=(m.K0[:M] if PURE else torch.cat([m.K0[:M],torch.zeros(M,4)],-1)).unsqueeze(0).expand(B,M,dk).clone(); Vm=torch.zeros(B,M,d); occ=torch.zeros(B,M); pk=torch.zeros(B,dk); usage=torch.zeros(B,M)
    ce_sum=0.0; ce_h=0.0; hits=tot=0; evict=0; present=0; conf_sum=0; mind=[]; seen={}; dq=[]; nm=[]
    namepos={(i,p) for i,sp in enumerate(spans) for s0,s1 in sp for p in range(s0,s1-1)}  # predicting byte p+1 from position p
    for t in range(L-1):
        h=m.gh(e[:,t],h); c=m.gc(e[:,t],c); xt=x[:,t]
        b=((xt==32)|(xt==10)|(xt==44)|(xt==46)|(xt==58)|(xt==63)|(xt==33)).float().unsqueeze(-1)
        logocc=torch.log(occ+1e-4); qv=torch.cat([c,torch.tanh(m.z(h))],-1)
        sim=torch.einsum("bd,bmd->bm",qv,K)/math.sqrt(d)
        if len(ARGV)>4 or Q.PUREADDR==3: sim=-torch.cdist(qv[:,:32].unsqueeze(1),K[:,:,:32]).squeeze(1)-0.0*torch.cdist(qv[:,32:].unsqueeze(1),K[:,:,32:]).squeeze(1); a=torch.softmax(sim/tau+torch.log(occ+1e-4),-1)
        else: a=torch.softmax((sim+(0 if PURE else logocc))/tau,-1)
        r=torch.einsum("bm,bmd->bd",a,Vm)
        # read accuracy at name positions: nearest embedding to r vs true next byte
        for i in range(B):
            # prefix string since last separator
            pstart=t
            while pstart>0 and x[i,pstart-1].item() not in (32,10,44,46,58,63,33): pstart-=1
            pref=bytes(x[i,pstart:t+1].tolist())
            if (i,t) in namepos:
                if (i,pref) in seen:
                    q0,K0_=seen[(i,pref)]; dq.append(((qv[i,:32]-q0[:32]).norm().item(), (qv[i,32:]-q0[32:]).norm().item()))
                    # was the slot where that write landed overwritten? compare current K at that slot's argmax to q0
                    nm.append((K[i]-q0).norm(dim=-1).min().item())
            if len(pref)>=1: seen.setdefault((i,pref),(qv[i].clone(),None))
            if (i,t) in namepos:
                tot+=1; tgt=x[i,t+1].item()
                cf=(a[i]*occ[i]).sum().reshape(1); lg=m.head(torch.cat([h[i],r[i],cf],-1)); ce_sum+=F.cross_entropy(lg[None],torch.tensor([tgt])).item()
                # value-as-logits: nearest-embedding read turned into a distribution (what a head would do with a clean value)
                lg2=m.head(torch.cat([h[i],m.E.weight[tgt],torch.ones(1)],-1)); ce_h+=F.cross_entropy(lg2[None],torch.tensor([tgt])).item()
                nn_=torch.cdist(r[i:i+1],m.E.weight).argmin().item(); hits+=int(nn_==tgt); conf_sum+=a[i].max().item()
                # is any slot holding a key within small distance of qv with value == e[tgt]? (binding present)
                dv=torch.cdist(Vm[i],m.E.weight[tgt:tgt+1]).squeeze(-1); dkq=torch.cdist(K[i],qv[i:i+1]).squeeze(-1)
                j=dv.argmin().item(); mind.append((round(dv[j].item(),2), round((K[i,j,:32]-qv[i,:32]).norm().item(),2), round((K[i,j,32:]-qv[i,32:]).norm().item(),2))); present+=int(dv[j].item()<0.5)
        wsim=torch.einsum("bd,bmd->bm",pk,K)/math.sqrt(d); mm=torch.softmax((wsim+logocc)/tau,-1)
        p=torch.sigmoid(wsim.max(-1,keepdim=True).values-m.theta); empty=torch.softmax(-occ/tau,-1)
        if len(ARGV)>3 or Q.PUREADDR==3:   # HARD eval-time allocation: match if c-part key dist<1 else LRU
            dmat=torch.cdist(pk[:,:32].unsqueeze(1),K[:,:,:32]).squeeze(1); jm=dmat.argmin(-1); ok=dmat.gather(1,jm[:,None]).squeeze(1)<1.0
            jf=(usage+m.ramp).argmin(-1); j=torch.where(ok,jm,jf); alpha=F.one_hot(j,M).float().unsqueeze(-1); usage=0.98*usage+alpha.squeeze(-1)+a
        elif Q.PUREADDR==2:
            mm2=torch.softmax(wsim/tau,-1); p2=torch.sigmoid((wsim.max(-1,keepdim=True).values-m.theta)/0.1); free=torch.softmax((-usage-m.ramp)/tau,-1); alpha=(p2*mm2+(1-p2)*free).unsqueeze(-1); usage=0.98*usage+alpha.squeeze(-1)+a
        else: alpha=torch.softmax(wsim/tau,-1).unsqueeze(-1) if PURE else (p*mm+(1-p)*empty).unsqueeze(-1)
        K=(1-alpha)*K+alpha*pk.unsqueeze(1); Vm=(1-alpha)*Vm+alpha*e[:,t].unsqueeze(1); occ=occ+alpha.squeeze(-1)*(1-occ)
        pk=qv; c=c*(1-b)
import statistics; print("re-mention vs first-mention query dist (c-part, z-part) median", [round(statistics.median([u[k] for u in dq]),2) for k in range(2)], "n", len(dq), "| min dist of any current key to first-mention key median", round(statistics.median(nm),2)); print("M",M,"best-value slot: median (value-dist, key-dist c-part, key-dist z-part)", [round(statistics.median([u[k] for u in mind]),2) for k in range(3)], "emb norm", round(m.E.weight.norm(dim=1).mean().item(),2)); print(f"dense-unseen name-CE via trained head {ce_sum/tot:.3f}; via head fed EXACT target embedding as r {ce_h/tot:.3f}"); print(f"name-byte positions {tot}: read-nearest==target {hits/tot:.3f}; binding present in memory {present/tot:.3f}; mean max-attn {conf_sum/tot:.2f}; theta {m.theta.item():.2f}")
