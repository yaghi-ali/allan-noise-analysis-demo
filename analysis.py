import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def savefig(fig,out):
    fig.tight_layout()
    fig.savefig(out/'preview.png',dpi=160,bbox_inches='tight')
    plt.close(fig)

def allan(y,dt,factors):
    y=np.asarray(y,float);f=np.asarray(factors)
    if y.ndim!=1 or len(y)<4 or not np.all(np.isfinite(y)) or not np.isfinite(dt) or dt<=0 or f.ndim!=1 or f.size==0 or not np.all(np.isfinite(f)) or np.any(f!=np.floor(f)) or np.any(f<1) or np.any(2*f>len(y)):raise ValueError('Invalid series, sample interval or averaging factors.')
    cs=np.r_[0.,np.cumsum(y)];values=[]
    for m in f.astype(int):
        avg=(cs[m:]-cs[:-m])/m;values.append(np.sqrt(.5*np.mean((avg[m:]-avg[:-m])**2)))
    return f*dt,np.array(values)
def example(out,seed):
    rng=np.random.default_rng(seed);n=32768;dt=.1;t=np.arange(n)*dt;white=rng.normal(0,1,n);drift=white+.002*t;f=np.unique(np.logspace(0,3,35).astype(int));tau,a=allan(white,dt,f);_,b=allan(drift,dt,f)
    fig,ax=plt.subplots(1,2,figsize=(10,4));ax[0].plot(t[::32],drift[::32],lw=.8);ax[0].set(xlabel='Time (s)',ylabel='Signal (a.u.)',title='White noise + drift')
    ax[1].loglog(tau,a,'o-',label='White noise');ax[1].loglog(tau,b,'o-',label='White + drift');ax[1].loglog(tau,1/np.sqrt(f),'k--',label='White expectation');ax[1].set(xlabel='Averaging time (s)',ylabel='Allan deviation (a.u.)');ax[1].legend();savefig(fig,out)
    np.savetxt(out/'allan.csv',np.column_stack([tau,a,b]),delimiter=',',header='tau_s,white_adev,white_drift_adev',comments='')
    return {'white_loglog_slope_first_15':float(np.polyfit(np.log(tau[:15]),np.log(a[:15]),1)[0]),'minimum_sampled_drift_adev':float(b.min()),'best_sampled_tau_s':float(tau[np.argmin(b)])}

def run(out='results',seed=42):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    metrics=example(out,seed);metrics.update(seed=seed,data='synthetic')
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2));return metrics
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='results');p.add_argument('--seed',type=int,default=42);a=p.parse_args();print(json.dumps(run(a.out,a.seed),indent=2))
