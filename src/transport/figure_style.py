import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
C=['#f9a48b','#5d5d2a','#7e0909','#C7C3AC','#DBA997']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,'axes.titlesize':11,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':C[1],'figure.facecolor':'white','axes.facecolor':'white','svg.fonttype':'none','legend.frameon':False,'savefig.dpi':220})

def lab(ax, label):
    ax.text(-.1, 1.05, label, transform=ax.transAxes, fontweight='bold', fontsize=12)

def draw_schematic(output):
    def save(fig, name):
        for ext in ['png', 'svg']:
            fig.savefig(output / (name + '.' + ext), bbox_inches='tight', facecolor='white')
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(10.4,4.3));ax.set(xlim=(0,10.4),ylim=(0,4.3));ax.axis('off')
    boxes=[(.15,2.65,2.5,1.0,'Full follicular reservoir\nH = capacity during 0–15 min',C[4]),(3.4,2.65,2.35,1.0,'Free intact tFNA\nin extracellular fluid',C[0]),(6.5,2.65,1.55,1.0,'Cell contact\nC_eff',C[3]),(8.8,2.65,1.4,1.0,'Intracellular\ntFNA',C[1])]
    for x,y,w,h,txt,col in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.03,rounding_size=.05',facecolor=col,edgecolor=C[1]));ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=9,color='white' if col==C[1] else '#222')
    def arrow(a,b,txt,xy,both=False):
        ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='<->' if both else '->',color=C[2],lw=1.8));ax.text(*xy,txt,fontsize=9,ha='center')
    arrow((2.7,3.15),(3.35,3.15),'wall crossing',(3.02,3.88),True)
    arrow((5.8,3.15),(6.45,3.15),'diffusive encounter',(6.1,3.88))
    arrow((8.1,3.15),(8.75,3.15),'uptake',(8.45,3.88))
    ax.text(.2,1.72,'0–15 min: ideal replenishment\nAfter 15 min: no external input',fontsize=10,color=C[2],linespacing=1.6)
    ax.text(3.5,1.55,'Tracer-based layer diffusion\naccounts for tissue resistance',fontsize=10,color=C[1])
    ax.text(.2,.35,'Initial tissue and cells are empty. Existing skin reservoirs continue to evolve after the mask is removed.',fontsize=9)
    ax.text(7.05,1.03,'Team CK limit: 1 µmol/L\ntFNA equivalent: 0.5 µmol/L',ha='center',fontsize=10,fontweight='bold')
    save(fig,'Fig02_Model_and_Window')
