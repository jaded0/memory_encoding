"""Classify a kick cell from its interval-average loss series (definitions: note section 1)."""
import numpy as np
def classify(its, loss, kick):
    its=np.array(its); loss=np.array(loss,float)
    sel=its>kick; its=its[sel]; loss=loss[sel]
    if len(loss)==0: return dict(cls='no data')
    n=len(loss); peak=float(np.nanmax(loss)); ipk=int(np.nanargmax(loss))
    tail=loss[int(n*0.8):]
    out=dict(peak=peak, peak_it=int(its[ipk]-kick), final=float(loss[-1]), tail_med=float(np.median(tail)))
    # recovery time: first window after peak starting run of 3 below 4
    rec=None
    for i in range(ipk,n-2):
        if np.all(loss[i:i+3]<4): rec=int(its[i]-kick); break
    out['recovery']=rec
    first6=np.where(loss>6)[0]
    out['first_exc']=int(its[first6[0]]-kick) if len(first6) else None
    # section-12 onset: 5 consecutive windows > 5, or one > 20 (>0 after kick)
    above5=loss>5; onset=None
    for i in range(n):
        if loss[i]>20 or (i+5<=n and above5[i:i+5].all()): onset=int(its[i]-kick); break
    out['onset']=onset
    if peak<=6: c='stable'
    elif np.all(tail<4) : c='transient'
    elif np.nanmax(loss)>100 and not np.any(loss[ipk:]<5): c='runaway'
    elif (out['first_exc'] or 0)>10000 and onset is not None and not np.all(tail<4): c='delayed collapse'
    elif onset is not None and out['first_exc']<=10000 and np.nanmin(loss[ipk:])>=4 or np.median(tail)>6: c='non-recovering'
    else: c='transient?'
    out['cls']=c; return out
