"""Memory localization only: strict read-time knockout at recall steps, no training.
No new decision rule: report every layer, and first-answer sensitivity to its fast read.
"""
import sys
import argparse
import json
from pathlib import Path
import torch

@torch.no_grad()
def run(a):
    sys.path.insert(0,a.code)
    from ephemeral_model import dfa_output_error
    from heldout import load_heldout_batches, update_mask
    from train import build_model,build_parser,positional_encoding
    from utils import initialize_charset,load_checkpoint,model_input,read_checkpoint,upgrade_legacy_config
    from metrics import recall_targets
    ck=read_checkpoint(a.checkpoint)
    c={**vars(build_parser().parse_args([])),**upgrade_legacy_config(ck['config'])}
    assert c.get('fast_read_norm','off') == 'off'
    assert not c.get('order2_layers','')
    charset,_,_,n=initialize_charset(c['dataset'])
    m=build_model(c,charset,n)
    m,_,_,_,_=load_checkpoint(a.checkpoint,m,c,device='cuda',checkpoint=ck)
    del ck
    m=m.cuda().eval()
    c['pe_matrix']=positional_encoding(c['positional_encoding_dim'],'cuda')
    batches=load_heldout_batches(c['dataset'],c['batch_size'],8,'cuda','test')
    result={}
    for knockout in ['none']+[f'L{i}' for i in range(len(m.linear_layers))]:
        first,all_hits=[],[]
        for texts,onehot in batches:
            targets=[recall_targets(t,c['dataset'])[0] for t in texts]
            mask=update_mask('strict',texts,c['dataset'],onehot)
            m.start_sequence_wipe()
            hidden=m.initHidden(len(texts))
            for t in range(onehot.shape[1]-1):
                rows=torch.tensor([t+1 in ts for ts in targets],device='cuda')
                handle=None
                if knockout != 'none' and bool(rows.any()):
                    layer=m.linear_layers[int(knockout[1:])]
                    def no_read(layer,inp,out):
                        fast=torch.where(layer.ephemeral_mask,layer.per_sample_weights,0)
                        read=torch.bmm(fast,inp[0].unsqueeze(-1)).squeeze(-1)
                        return out-read*rows[:,None]
                    handle=layer.register_forward_hook(no_read)
                out,hidden=m(model_input(onehot,t,c['input_mode'],c['pe_matrix']),hidden)
                if handle:
                    handle.remove()
                for b,ts in enumerate(targets):
                    if t+1 in ts:
                        hit=int(out[b].argmax()) == int(onehot[b,t+1].argmax())
                        all_hits.append(hit)
                        if t+1 == min(ts):
                            first.append(hit)
                _,err=dfa_output_error(out,onehot[:,t+1],torch.nn.CrossEntropyLoss(reduction='none'))
                m.fast_only_dfa_step(err*mask[:,t,None],c['learning_rate'],c['ephemeral_update_clamp'],c.get('grad_norm_clip',0))
        result[knockout]={'first':sum(first)/len(first),'recall':sum(all_hits)/len(all_hits)}
    Path(a.out).write_text(json.dumps(result,indent=2))
    print(a.out,result,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--code',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--out',required=True)
    torch.set_num_threads(4)
    run(p.parse_args())
