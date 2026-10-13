"""Finite follicular reservoir, time-limited application, and cell-contact exposure.

Mass states use ng TH21-core DNA equivalent/cm2. Source and uptake are counters.
No formulation-release reservoir, direct stratum-corneum route, or intact cell export.
The auxiliary S state supports calibration against the published source experiment.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix
from physics import derived, CORE_MW

class Model:
    def __init__(self,p):
        self.p=p;self.d=derived(p);d=self.d
        self.edges=np.linspace(p.sc_um,p.bottom_um,round((p.bottom_um-p.sc_um)/p.dx_um)+1)
        self.z=(self.edges[:-1]+self.edges[1:])/2;self.dx=np.diff(self.edges)*1e-4;n=len(self.z)
        self.he=np.linspace(0,p.follicle_length_um,round(p.follicle_length_um/p.follicle_dx_um)+1)
        self.hz=(self.he[:-1]+self.he[1:])/2;self.hdx=np.diff(self.he)*1e-4;nh=len(self.hz)
        self.S=0;i=1;self.H=np.arange(i,i+nh);i+=nh
        self.E=np.arange(i,i+n);i+=n;self.I=np.arange(i,i+n);i+=n
        self.F=np.arange(i,i+3);i+=3;self.O=i;i+=1;self.mass_n=i
        self.A=i;i+=1;self.U=np.arange(i,i+n);i+=n;self.size=i
        self.layer=(self.z>=p.ve_bottom_um).astype(int)
        self.eps=np.where(self.layer==0,p.epsilon_ve,p.epsilon_dermis);self.vol=self.eps*self.dx
        self.D=np.array([d['VE']['D_cm2_h'],d['dermis']['D_cm2_h']])[self.layer]
        self.K=np.array([d['VE']['partition'],d['dermis']['partition']])[self.layer]
        self.hvol=d['follicle_volume_cm3_cm2']*self.hdx/self.hdx.sum()
        self.hcap=d['follicle_capacity_ng_cm2']*self.hdx/self.hdx.sum()
        self.ku=np.full(n,p.internalization_hazard_h)
        self.alpha=np.ones(n)
        self.gface=np.zeros(n-1)
        good=(self.D[:-1]*self.K[:-1]>0)&(self.D[1:]*self.K[1:]>0)
        self.gface[good]=1/(self.dx[:-1][good]/(2*self.eps[:-1][good]*self.D[:-1][good]*self.K[:-1][good])+self.dx[1:][good]/(2*self.eps[1:][good]*self.D[1:][good]*self.K[1:][good]))
        self.wall=np.zeros((nh,n))
        for h in range(nh):
            overlap=np.maximum(0,np.minimum(self.edges[1:],self.he[h+1])-np.maximum(self.edges[:-1],self.he[h]))
            self.wall[h]=d['wall_P_cm_h']*d['wall_area_cm2_cm2']*overlap/p.follicle_length_um
        self.gentry=0 if p.follicles_cm2==0 else d['follicle_D_cm2_h']*d['entry_area_fraction']/(d['entry_length_um']*1e-4+self.hdx[0]/2)
        self.gh=d['follicle_D_cm2_h']*(d['follicle_volume_cm3_cm2']/(p.follicle_length_um*1e-4))/(self.hdx[:-1]/2+self.hdx[1:]/2)
        self.gexit=0 if self.D[-1]==0 else self.eps[-1]*self.D[-1]/(self.dx[-1]/2)
        self.jac=self.sparsity()

    def sparsity(self):
        s=lil_matrix((self.size,self.size),dtype=int)
        for i in range(self.size):s[i,i]=1
        def edge(a,b):
            for x in np.atleast_1d(a):
                for y in np.atleast_1d(b):s[x,y]=1;s[y,x]=1
        edge(self.S,self.H[0])
        for a,b in zip(self.H[:-1],self.H[1:]):edge(a,b)
        for a,b in zip(self.E[:-1],self.E[1:]):edge(a,b)
        for h,j in zip(*np.where(self.wall>0)):edge(self.H[h],self.E[j])
        for j in range(len(self.z)):
            edge(self.E[j],self.I[j]);s[self.U[j],self.E[j]]=1
        for f in self.F:s[f,:self.mass_n]=1
        s[self.O,:self.mass_n]=1;s[self.A,:self.mass_n]=1
        return s.tocsr()

    def rhs(self,t,y,protocol='ideal',feeding=False):
        p=self.p;dy=np.zeros_like(y);v=np.maximum(y,0)
        def flux(a,b,f):dy[a]-=f;dy[b]+=f
        c=v[self.E]/self.vol;q=np.divide(c,self.K,out=np.zeros_like(c),where=self.K>0)
        if p.follicles_cm2>0:
            H=v[self.H];empty=np.maximum(0,1-H/self.hcap);hc=H/self.hvol
            if protocol=='source_experiment':
                # Published 24 h source experiment: 100 um finite mobile donor; no mask cutoff.
                flux(self.S,self.H[0],self.gentry*(v[self.S]/.01*empty[0]-hc[0]))
            elif not feeding:
                # No new entry after mask removal; the open lumen can lose material outward.
                flux(self.H[0],self.O,self.gentry*hc[0])
            f=self.gh*(hc[:-1]*empty[1:]-hc[1:]*empty[:-1]);dy[self.H[:-1]]-=f;dy[self.H[1:]]+=f
            f=self.wall*(hc[:,None]-q[None,:]*empty[:,None]);dy[self.H]-=f.sum(axis=1);dy[self.E]+=f.sum(axis=0)
        f=self.gface*(q[:-1]-q[1:])
        dy[self.E[:-1]]-=f;dy[self.E[1:]]+=f
        f=self.ku*v[self.E];dy[self.E]-=f;dy[self.I]+=f;dy[self.U]+=f
        kd=np.log(2)/p.intact_half_h;ki=kd
        for ids,dest,k in [(np.array([self.S]),self.F[0],kd),(self.H,self.F[1],kd),(self.E,self.F[2],kd),(self.I,self.F[2],ki)]:
            loss=k*v[ids];dy[ids]-=loss;dy[dest]+=loss.sum()
        flux(self.E[-1],self.O,self.gexit*c[-1])
        if feeding and protocol=='ideal':
            # Dirichlet reservoir at its specified fill level: explicitly account for replacement.
            need=-dy[self.H].copy();dy[self.A]+=np.maximum(need,0).sum();dy[self.O]+=np.maximum(-need,0).sum();dy[self.H]=0.
        return dy

    def initial(self,protocol='ideal'):
        y=np.zeros(self.size)
        if protocol=='source_experiment':y[self.S]=100.
        else:y[self.H]=self.hcap*self.p.loading_fraction
        return y

    def solve(self,times,protocol='ideal',rtol=2e-7,method='BDF',y0=None):
        times=np.asarray(times,float);y0=self.initial(protocol) if y0 is None else y0.copy()
        scale=max(float(y0[:self.mass_n].sum()),1.)
        opts=dict(method=method,rtol=rtol,atol=max(scale*1e-12,1e-12),jac_sparsity=self.jac)
        end=times[-1];cut=self.p.application_min/60
        if protocol!='ideal' or cut<=0:
            sol=solve_ivp(lambda t,y:self.rhs(t,y,protocol,False),(0,end),y0,t_eval=times,**opts)
            if not sol.success:raise RuntimeError(sol.message)
            return sol.y.T
        first=times[times<=min(cut,end)]
        eval1=np.unique(np.r_[first,min(cut,end)])
        s1=solve_ivp(lambda t,y:self.rhs(t,y,protocol,True),(0,min(cut,end)),y0,t_eval=eval1,**opts)
        if not s1.success:raise RuntimeError(s1.message)
        out=s1.y[:,np.searchsorted(eval1,first)].T
        if end>cut:
            second=times[times>cut]
            s2=solve_ivp(lambda t,y:self.rhs(t,y,protocol,False),(cut,end),s1.y[:,-1],t_eval=second,**opts)
            if not s2.success:raise RuntimeError(s2.message)
            out=np.vstack([out,s2.y.T])
        return out

    def contact(self,Y):
        return np.asarray(Y)[...,self.E]/self.vol/CORE_MW*self.alpha

    def outputs(self,times,Y):
        import pandas as pd
        total=Y[:,self.E]+Y[:,self.I];cs=self.contact(Y)
        r={'time_h':times,'time_min':np.asarray(times)*60,'follicle_ng_cm2':Y[:,self.H].sum(axis=1),
           'additional_input_ng_cm2':Y[:,self.A],'intracellular_ng_cm2':Y[:,self.I].sum(axis=1),
           'uptake_cumulative_ng_cm2':Y[:,self.U].sum(axis=1),'fragments_ng_cm2':Y[:,self.F].sum(axis=1),
           'outside_ng_cm2':Y[:,self.O],'max_contact_uM':cs.max(axis=1)}
        for mask,name in [(self.layer==0,'VE'),(self.layer==1,'dermis')]:
            r[name+'_ng_cm2']=total[:,mask].sum(axis=1)
            r[name+'_total_uM']=r[name+'_ng_cm2']/self.dx[mask].sum()/CORE_MW
            r[name+'_free_uM']=Y[:,self.E][:,mask].sum(axis=1)/self.vol[mask].sum()/CORE_MW
            r[name+'_contact_uM']=np.average(cs[:,mask],weights=self.vol[mask],axis=1)
            r[name+'_intracellular_ng_cm2']=Y[:,self.I][:,mask].sum(axis=1)
            r[name+'_uptake_ng_cm2']=Y[:,self.U][:,mask].sum(axis=1)
            r[name+'_covered_fraction']=np.average(cs[:,mask]>=self.p.target_uM,weights=self.dx[mask],axis=1)
        r['intact_ng_cm2']=r['follicle_ng_cm2']+r['VE_ng_cm2']+r['dermis_ng_cm2']+Y[:,self.S]
        r['DNA_remaining_ng_cm2']=r['intact_ng_cm2']+r['fragments_ng_cm2']
        r['mass_minus_input_ng_cm2']=Y[:,:self.mass_n].sum(axis=1)-Y[:,self.A]
        return pd.DataFrame(r)

    def optical(self,y,z):
        tissue=y[self.E]+y[self.I]
        j=np.clip(np.searchsorted(self.edges,z,side='right')-1,0,len(self.z)-1)
        v=tissue[j]/(self.dx[j]*1e4);v=np.where((z>=self.p.sc_um)&(z<=self.p.bottom_um),v,0)
        h=np.clip(np.searchsorted(self.he,z,side='right')-1,0,len(self.hz)-1)
        return v+np.where((z>=0)&(z<self.p.follicle_length_um),y[self.H][h]/(self.hdx[h]*1e4),0)

def intervals_above(t,v,target):
    """All continuous intervals in a piecewise-linear series, with exact crossings."""
    t=np.asarray(t,float);v=np.asarray(v,float);above=v>=target
    starts=np.flatnonzero(above&~np.r_[False,above[:-1]])
    ends=np.flatnonzero(above&~np.r_[above[1:],False]);segments=[]
    for i,j in zip(starts,ends):
        a=t[0] if i==0 else t[i-1]+(t[i]-t[i-1])*(target-v[i-1])/(v[i]-v[i-1])
        b=t[-1] if j==len(t)-1 else t[j]+(t[j+1]-t[j])*(target-v[j])/(v[j+1]-v[j])
        if b-a>1e-12:segments.append([float(a),float(b)])
    return segments

def best_window(t,v,width):
    """Maximize the minimum concentration over a continuous interval of fixed width."""
    t=np.asarray(t,float);v=np.asarray(v,float)
    if t[-1]-t[0]<width:return dict(value_uM=float('nan'),start_min=float('nan'),end_min=float('nan'))
    lo=float(v.min());hi=float(v.max());start=float(t[0])
    for _ in range(55):
        mid=(lo+hi)/2;segments=intervals_above(t,v,mid)
        eligible=[a for a,b in segments if b-a>=width-1e-10]
        if eligible:lo=mid;start=eligible[0]
        else:hi=mid
    return dict(value_uM=lo,start_min=start,end_min=start+width)
