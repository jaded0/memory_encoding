"""Slow-weight features of checkpoints: slow matrix = batch-mean per_sample_weights with fast positions zeroed.
usage: python features.py out.json ckpt1 ckpt2 ... (name = file path)"""
import torch,sys,json
out={}
for f in sys.argv[2:]:
    c=torch.load(f,map_location='cpu',weights_only=False); sd=c['model_state_dict']; r={'iter':c['iter']}
    for l in range(3):
        S=sd[f'linear_layers.{l}.per_sample_weights'].float().mean(0)*(1-sd[f'linear_layers.{l}.ephemeral_mask'].float())
        r[f'sv_t{l}']=float(torch.linalg.matrix_norm(S,2)); r[f'fro_t{l}']=float(torch.linalg.matrix_norm(S))
    o=sd['i2o.per_sample_weights'].float().mean(0)
    r['i2o_fro']=float(torch.linalg.matrix_norm(o)); r['i2o_sv']=float(torch.linalg.matrix_norm(o,2)); r['i2o_w_fro']=float(sd['i2o.weight'].float().norm())
    r['bias_norm']=float(sum(sd[f'linear_layers.{l}.bias'].float().norm()**2 for l in range(3))**0.5)
    out[f]=r; print(f,r,flush=True)
json.dump(out,open(sys.argv[1],'w'))
