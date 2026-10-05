import re,sys
def parse(path):
    t=open(path,errors='replace').read()
    t=re.sub(r'\x1b\[[0-9;]*m','',t)
    blocks=re.split(r'--- Interval metrics \(ending @ iter (\d+)[^\n]*\n',t)[1:]
    out={}
    for i in range(0,len(blocks),2):
        it=int(blocks[i]); b=blocks[i+1]
        d={}
        for k in ['loss','recall_loss','other_loss','recall_acc']:
            m=re.search(r'^\s*'+k+r': ([\d.eE+-]+|nan|inf)',b,re.M)
            if m: d[k]=float(m.group(1))
        out[it]=d
    return out
if __name__=='__main__':
    o=parse(sys.argv[1]); step=int(sys.argv[2]) if len(sys.argv)>2 else 1
    for it in sorted(o):
        if it%step==0: print(it,o[it])
