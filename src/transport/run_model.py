from pathlib import Path
import json
import argparse
import pandas as pd
from dataclasses import replace
import numpy as np
from physics import Inputs
from model import Model
BASE=Path(__file__).resolve().parents[2]
DATA=BASE/'data'/'transport'
parser=argparse.ArgumentParser(description='Run the TH21 tFNA skin transport model.')
parser.add_argument('--output-dir', type=Path, default=BASE/'results'/'transport'/'generated')
args=parser.parse_args()
OUT=args.output_dir.resolve()
reference=BASE/'results'/'transport'
protected=[BASE/name for name in ('data','models','logs')]+[reference/'images']
if OUT == reference.resolve() or any(OUT.is_relative_to(path.resolve()) for path in protected):
 raise ValueError('Choose a separate output directory outside packaged data, models, logs, and reference images.')
OUT.mkdir(parents=True,exist_ok=True)
p=Inputs(**json.loads((DATA/'reference_inputs.json').read_text(encoding='utf-8')))
m=Model(p)
mins=np.unique(np.r_[np.arange(0,15.001,.1),np.arange(16,1441,1),np.arange(1450,6001,10)])
y=m.solve(mins/60);o=m.outputs(mins/60,y);o.to_csv(OUT/'timeseries.csv',index=False)
idx=np.searchsorted(mins,15);a=o.iloc[idx];dose=m.d['follicle_capacity_ng_cm2']+a.additional_input_ng_cm2
ret=[]
for minute in [15,60,240]:
 r=o.iloc[np.searchsorted(mins,minute)]
 for layer in ['follicle','VE','dermis']:
  amount=r[layer+'_ng_cm2'];ret.append(dict(time_min=minute,layer=layer,mass=amount,percent=amount/dose*100))
clear={}
for frac in [.1,.05]:
 v=o.intact_ng_cm2.to_numpy()[idx:];t=mins[idx:];target=v[0]*frac;k=np.flatnonzero(v<=target)[0]
 hit=t[k-1]+(t[k]-t[k-1])*(target-v[k-1])/(v[k]-v[k-1]);clear[str(frac)]=(hit-15)/60
summary=dict(at15=a.to_dict(),dose_ug=dose/1000,retention=ret,clearance=clear,peaks={layer:float(o[layer+'_contact_uM'].max()) for layer in ['VE','dermis']})
checks=dict(mass_relative=float(abs(o.mass_minus_input_ng_cm2-m.d['follicle_capacity_ng_cm2']).max()/m.d['follicle_capacity_ng_cm2']),min_state=float(y.min()),input_after_removal=float(np.ptp(o.additional_input_ng_cm2.to_numpy()[idx:])))
if not (0 <= checks['mass_relative']<1e-7 and checks['min_state']>-1e-7 and 0 <= checks['input_after_removal']<1e-8):
 raise RuntimeError(f'Physical checks failed: {checks}')
summary['checks']=checks
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))

# Retain the latest figure style and regenerate time-dependent panels consistently.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import figure_style as pf
images=OUT/'images';images.mkdir(exist_ok=True)
C=pf.C
def save(fig,name):
 fig.savefig(images/name,bbox_inches='tight',facecolor='white',dpi=240)
 fig.savefig((images/name).with_suffix('.svg'),bbox_inches='tight',facecolor='white')
 plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(10.3,3.7),layout='constrained')
for ax,layer,title in zip(axs,['VE','dermis'],['Viable epidermis','Shallow dermis']):
 q=o[o.time_min<=15];ax.plot(q.time_min,q[layer+'_contact_uM'],c=C[2],lw=2,label='Cell-contact tFNA concentration')
 ax.text(.05,.65,f"At 15 min\n{a[layer+'_contact_uM']:.4f} µmol/L",transform=ax.transAxes,fontsize=10)
 ax.text(.98,.96,'Team tFNA limit: 0.5 µmol/L (off scale)',transform=ax.transAxes,ha='right',va='top',fontsize=8,color=C[1])
 ax.set(xlabel='Time during application (min)',ylabel='Intact tFNA at cell surface (µmol/L)',xlim=(0,15),ylim=(0,.08),yticks=np.arange(0,.081,.02),xticks=[0,5,10,15],title=title)
axs[1].legend(fontsize=8,loc='upper left',bbox_to_anchor=(.02,.88));pf.lab(axs[0],'A');pf.lab(axs[1],'B')
save(fig,'Fig03_Application_Exposure.png')
from plot_depth_heatmap import draw_depth_and_exposure
heatmap_mask=mins<=240
fig=draw_depth_and_exposure(m,mins[heatmap_mask]/60,y[heatmap_mask],o)
save(fig,'Fig04_Depth_and_Delayed_Exposure_Heatmap.png')
heatmap_contact=np.maximum(m.contact(y[heatmap_mask]),0)
pd.DataFrame({
    'time_h':np.repeat(mins[heatmap_mask]/60,len(m.z)),
    'depth_um':np.tile(m.z,heatmap_mask.sum()),
    'free_uM':heatmap_contact.ravel(),
}).to_csv(OUT/'depth_time_heatmap.csv',index=False)
fig,axs=plt.subplots(1,2,figsize=(10.3,3.7),layout='constrained');ax=axs[0];q=o[o.time_min<=15]
ax.plot(q.time_min,q.uptake_cumulative_ng_cm2,c=C[1],ls='--',lw=2,label='Cumulative cellular entry');ax.plot(q.time_min,q.intracellular_ng_cm2,c=C[2],lw=2,label='Surviving intracellular carrier');ax.set(xlabel='Time during application (min)',ylabel='tFNA core DNA (ng/cm²)',title='Uptake is an extracellular sink',xlim=(0,15),xticks=[0,5,10,15]);ax.legend(fontsize=8);pf.lab(ax,'A')
val=[]
for rate in [0,p.internalization_hazard_h,1,4]:
 sm=Model(replace(p,internalization_hazard_h=rate));sy=sm.solve([0,.25]);val.append(sm.outputs([0,.25],sy).VE_contact_uM.iloc[-1])
ax=axs[1];ax.bar(['0','0.077','1','4'],val,color=[C[3],C[2],C[0],C[4]],edgecolor=C[1]);ax.set(xlabel='Internalization rate (h⁻¹)',ylabel='Cell-contact tFNA at 15 min (µmol/L)',title='Faster uptake reduces available tFNA');pf.lab(ax,'B');save(fig,'Fig05_Uptake_and_Contact.png')
pf.draw_schematic(images)

# Explicit tables for the final report time points, with units in column names.
pd.DataFrame(ret).rename(columns={'mass':'intact_ng_cm2','percent':'percent_total_input'}).to_csv(OUT/'layer_retention.csv',index=False)
profiles=[]
for minute in [15,60,240]:
    k=np.searchsorted(mins,minute)
    for depth,concentration in zip(m.z,m.contact(y[k])):
        profiles.append(dict(time_min=minute,depth_um=float(depth),free_uM=float(concentration)))
pd.DataFrame(profiles).to_csv(OUT/'depth_profiles.csv',index=False)
pd.DataFrame({'internalization_hazard_h':[0,p.internalization_hazard_h,1,4],'VE_contact_uM_at15':val}).to_csv(OUT/'uptake_sensitivity.csv',index=False)
(OUT/'reference_inputs.json').write_text(json.dumps(vars(p),indent=2),encoding='utf-8')
print('Model run and physical checks passed.')
