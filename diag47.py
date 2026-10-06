import sys, random, math, torch
sys.argv=["x"]; import arch_vet_p47 as P, arch_vet_p46 as T
seed=2; torch.manual_seed(seed); rng=random.Random(seed)
m=P.SBC(16,4); opt=torch.optim.Adam(m.parameters(),lr=3e-3)
er=random.Random(999); ev=[T.make_batch(er,16,(2,4),2,4,False) for _ in range(6)]
def probe(m,tau):
    m.eval(); 
    with torch.no_grad():
        bt=ev[0]; x=bt["x"]; B,L=x.shape; d=16; M=4
        # replicate forward capturing addresses
        e=m.E(x); h=torch.zeros(B,d); K=m.K0[:M].unsqueeze(0).expand(B,M,d).clone(); Vm=torch.zeros(B,M,d); occ=torch.zeros(B,M); prev=torch.zeros(B,d)
        wr=[[] for _ in range(B)]; rd=[[] for _ in range(B)]
        for t in range(L):
            h=m.ctrl(e[:,t],h); win=torch.cat([prev,e[:,t]],-1); prev0=prev; prev=e[:,t]
            logocc=torch.log(occ+1e-4); q=m.qa[0]*e[:,t]+m.qa[1]*prev0
            sim=torch.einsum("bd,bmd->bm",q,K)/4; a=torch.softmax((sim+logocc)/tau,-1)
            k=m.ka[0]*e[:,t]+m.ka[1]*prev0; v=m.v(win); g=torch.sigmoid(m.g(h))
            wsim=torch.einsum("bd,bmd->bm",k,K)/4; mm=torch.softmax((wsim+logocc)/tau,-1)
            p=torch.sigmoid(wsim.max(-1,keepdim=True).values-m.theta); empty=torch.softmax(-occ/tau,-1)
            addr=p*mm+(1-p)*empty; alpha=(g*addr).unsqueeze(-1)
            for b in range(B):
                tok=x[b,t].item()
                if T.VAL0<=tok<T.FIL0 and x[b,t-1].item()>=T.KEY0 and x[b,t-1].item()<T.VAL0: wr[b].append((x[b,t-1].item()-T.KEY0, addr[b].argmax().item(), round(g[b].item(),2)))
                if t>0 and x[b,t-1].item()==T.Q: rd[b].append((tok-T.KEY0, a[b].argmax().item(), round(a[b].max().item(),2)))
            K=(1-alpha)*K+alpha*k.unsqueeze(1); Vm=(1-alpha)*Vm+alpha*v.unsqueeze(1); occ=occ+alpha.squeeze(-1)*(1-occ)
    m.train(); return wr[0],rd[0],wr[1],rd[1]
for s in range(3001):
    tau=1.0*(0.1)**(s/3000); gmin=0.2*(1-s/3000)
    bt=T.make_batch(rng,16,(2,4),2,4,False)
    logits,gates,occ=m(bt["x"],tau=tau,states=True,gmin=gmin)
    loss=T.ce_loss(logits,bt["x"],bt["ans"]); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    if s%500==0:
        print(s, "acc", round(P.acc(m,ev,tau=max(tau,0.05)),3), "theta", round(m.theta.item(),2), "ka", [round(x,2) for x in m.ka.mean(1).tolist()], "qa", [round(x,2) for x in m.qa.mean(1).tolist()], probe(m,tau), flush=True)
