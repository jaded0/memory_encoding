# Generates cell lists. Line: name ckpt_tag n_iters m print_freq
M=[0.3,0.5,1,1.5,2,3]
def e1(lin,stage,start,horizon,pf):
    return [(f"{lin}{stage//1000 if stage%1000==0 else stage/1000}k_m{m}".replace('.','p').replace('p0k','k') ,f"{lin}_{start:08d}",start+horizon,m,pf) for m in M]
cells={}
cells['O1']=e1('X1a',2500,2500,20000,250)+e1('X1a',5000,5000,20000,250)
cells['O2']=e1('X1a',10000,10000,20000,250)+e1('X1a',20000,20000,20000,250)
cells['O3']=e1('B',40000,40000,25000,500)+e1('B',80000,80000,25000,500)
e2=lambda lin,s,ms:[(f"E2_{lin}{s//1000}k_m{str(m).replace('.','p')}",f"{lin}_{s:08d}",s+50000,m,500) for m in ms]
cells['O4']=e2('B',180000,[0.5,0.3,0.1])+e2('B',205000,[0.5,0.3])
cells['O2']+=e2('X3',230000,[0.5,0.3,0.1])
cells['D1']=e1('B',150000,150000,50000,500)
cells['D2']=e1('B',120000,120000,50000,500)
if __name__=='__main__':
    for k,v in cells.items():
        open(f'cells_{k}.txt','w').write(''.join(' '.join(map(str,c))+'\n' for c in v))
        print(k,len(v),sum(c[2]-int(c[1].split('_')[1]) for c in v))
