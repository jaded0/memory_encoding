"""Build the final vault note from completed immutable result files (manticore only)."""
import json
from pathlib import Path
import numpy as np
from measure import analyze

BASE=Path('/home/jaden/p6_look_2026-11-05')
RESULTS=BASE/'results'
TITLE='ephemeral weights P6 write concentration look 2026-11-05'
NOTE=Path('/home/jaden/Documents/brain')/(TITLE+'.md')

def read(name):
    return json.loads((RESULTS/(name+'.json')).read_text())

def decision(d):
    share,j=d['store']['median'],d['jaccard']['mean']
    if share < .5 and j < 3*d['baseline']:
        return 'Worth running P6'
    if share > .8 or j > .5:
        return 'Not worth it'
    return 'Ambiguous'

def fmt(x):
    return f'{x:.3f}'

def dist(d):
    return f"{fmt(d['median'])} [{fmt(d['q25'])}–{fmt(d['q75'])}]"

def rng(values):
    return fmt(values[0]) if len(values)==1 else f'{min(values):.3f}–{max(values):.3f}'

def pool(names,task):
    runs=[read(n) for n in names]
    layers={}
    if len(names)==1:
        return runs[0]['layers'],runs
    for layer in runs[0]['layers']:
        ds=[]
        for n in names:
            z=np.load(RESULTS/(n+'.'+layer+'.npz'))
            ds.append({k:z[k] for k in z.files})
        maxT=max(d['E'].shape[1] for d in ds)
        def pad(x):
            return np.pad(x,[(0,0),(0,maxT-x.shape[1])]+([(0,0)] if x.ndim==3 else []))
        E,A,valid=[np.concatenate([pad(d[k]) for d in ds]) for k in ('E','A','valid')]
        stores=[list(np.flatnonzero(row)) for d in ds for row in d['store_mask']]
        layers[layer]=analyze(E,A,valid,stores,task)
        # Do not interpret pooled neuron IDs as identities across training seeds.
        for variant in ('energy','activation'):
            for k in ('16','64','256'):
                q=layers[layer]['selections'][variant][k]
                q['ever_selected_by_seed']=[r['layers'][layer]['selections'][variant][k]['ever_selected'] for r in runs]
                q['top5_share_by_seed']=[r['layers'][layer]['selections'][variant][k]['top5_share'] for r in runs]
                for key in ('counts','ever_selected','top5_share','selection_count'):
                    q.pop(key,None)
    return layers,runs

def main():
    specs=[('KR plain rms',[f'kr_plain_rms_s{s}' for s in (2718,3141,1618)],'kr'),
           ('KR plain ln',[f'kr_plain_ln_s{s}' for s in (2718,1414,1618)],'kr'),
           ('KR DC2 ln',[f'kr_DC2_ln_s{s}' for s in (2718,3141,1618)],'kr')]
    specs += [(f'MQAR K{k}',[f'mqar_K{k}'],'mqar') for k in (4,8,16,32)]
    groups={}
    allruns=[]
    for label,names,task in specs:
        layers,runs=pool(names,task)
        groups[label]=dict(layers=layers,runs=runs,task=task,names=names)
        allruns.extend(runs)
    probes={k:read(f'probe_K{k}') for k in (4,8,16,32)}
    (BASE/'pooled_summary.json').write_text(json.dumps(groups,indent=2))
    verdicts=[]
    for label,g in groups.items():
        ls=("L1","L2") if g["task"]=="kr" else ("L0",)
        cells=[]
        for l in ls:
            q=g["layers"][l]["selections"]["energy"]["64"]
            cells.append(f"{l} **{decision(q)}** (top64 share {q['store']['median']:.3f}; mean J {q['jaccard']['mean']:.3f})")
        verdicts.append(label+": "+", ".join(cells))
    text=['---',f'title: "Ephemeral weights: P6 write concentration cheap look"','date: 2026-11-05',
          'tags: [ephemeral-weights, research, analysis]','project: ephemeral-weights','status: final',
          'reading: unread','reading_priority: 2','reading_est: 12',
          'reading_focus: "KR: ambiguous (overlap too high); MQAR K4/8/16/32: not worth it in recall-supporting L0 (top64 already holds >80% of write energy)."',
          '---','', '# P6: write concentration cheap look, 2026-11-05','',
          '> [!summary] Fixed-rule verdicts — analysis only; no authorization to train']
    text += ['> - '+v for v in verdicts]
    text += ['', 'The layer-level distinctions are intentional: there is no new rule to average layers into a favorable task verdict. Key recall uses L1 and L2 as specified in [[ephemeral weights cheap checks 2026-10-16]]; MQAR read-time knockouts below localize net beneficial recall dependence to **L0**, not the diffuse deeper layers. Activation selection is evaluated alongside energy selection, not substituted after seeing results.',
             '', '## Checkpoints and provenance','',
             'All checkpoint/data commands, copies, analysis and vault writes ran through **manticore** (only the optional omen directory check was local). GPU1 was free of compute at the start; no training or deckard GPU use. The s1414 replacement ran on CPU. Source checkpoint files on deckard were read-only. Eight full held-out batches (**128 episodes/checkpoint**) in stored order: KR **validation**, MQAR **test**. Slow parameters frozen, each episode’s fast state wiped, **observed** protocol writes after every real target. Test/validation distinction matters: the pre-existing K13 analysis used validation, not this test panel.',
             '', '- KR actual trained code: **b26e88dc25ee829b93b65f59f678a0acc87c4385**, named in BFR note and `scripts/run_kr.sh`, replay archive `manticore:~/p6_look_2026-11-05/code_bfr`. No per-run `code/COMMIT` exists in these archived paths.',
             '- MQAR actual trained code: **983ca8be8d7713b8f4c7f8f52fb74eb90017a515**, consistently named by all four `code_commit`/`pilot_run.json` files. **Not 0f512f8 as in the brief**: use the actual training archive `~/p6_look_2026-11-05/code_k13`. All checkpoint loads reported matching config. This is a provenance correction, not retraining or a changed forward.',
             '- **Health correction:** initial assumed plain-ln seed3141 has prior 2048-episode strict **.94140625**, not ~1.000 (s5772=.93603515625). Primary healthy plain-ln seeds are **2718, 1414, 1618**, each prior strict **1.000**. Initial s3141 measurements are retained only as sensitivity data and excluded from primary tables; protocol has a dated addendum. Plain-rms and DC2-ln use **2718, 3141, 1618**, all current observed recall 1.000. No concentration-based checkpoint filtering.',
             '- **Optional additive_dense skipped:** no finished checkpoint found in the local C6/result locations checked on manticore or omen; C6b source checkpoints live on deckard/ORC. No ORC pull and no optional dense analysis claimed.',
             '', 'Exact checkpoint paths and SHA256 (all on manticore unless explicitly deckard):','',
             '| arm / dataset | iteration | observed recall | checkpoint | SHA256 |', '|---|---:|---:|---|---|']
    for r in allruns:
        text.append(f"| {r['dataset']} / {r['config']['seed']} / {r['config']['fast_read_norm']} / DC={r['config']['decouple_fast_feedback'] or 'off'} | {r['iteration']} | {r['observed_recall']:.4f} | `{r['checkpoint']}` | `{r['checkpoint_sha256']}` |")
    text += ['', 'For MQAR K∈{4,8,16,32}, original source is `deckard:/home/jaden/k13b_2026-10-28/pilot/mqar2_<K>_q4-lin-s2718/latest_checkpoint.pth`; copied destination is `manticore:/home/jaden/p6_look_2026-11-05/checkpoints/K<K>/latest_checkpoint.pth`. **All four source/destination SHA256 pairs match.** Datasets were copied from `deckard:~/k13b_2026-10-28/code/synth_datasets/mqar2_<K>_q4/`, all copied data files hash-verified against the source (`datasets.sha256`, `datasets_verify.txt`). KR uses the original BFR dataset. `deckard.sha256` and `copied_metadata.sha256` retain the checkpoint source and metadata hashes.',
             '', '## Definitions and fixed decision rule','',
             r'For every step and fast layer, **Eᵢ = pᵢ² · Σⱼ mᵢⱼ xⱼ²**. `p` is read from the training code’s `fast_write_projected`; DC2 L2 therefore uses **B_fast**, not the slow DFA feedback. This is squared Frobenius row energy of the **unclamped proposed write**, omitting common (lr·α)² and forgetting factors. KR weight clamp 1 may alter the realized update: this analysis is the requested formula, not a post-clamp measurement.',
             '', '- **M1:** retained fraction of total E for k=16,64,256, overall and store only, plus normalized participation `(ΣE)²/(N·ΣE²)`. Median and IQR are over individual real steps/store steps, not over medians. Zero-energy steps excluded from energy shares/PR and counted below; padding excluded everywhere.',
             '- **Selection:** primary sorts descending **E** (the brief’s “|p|-energy”), not bare |p|. Variant sorts absolute **post-nonlinearity activation**, GELU for trunk / tanh for i2h; the resulting share still measures **E**, not activation mass. PR is energy-based and identical for both selectors. Stable neuron-index tie-break; ties/zeros can matter especially in L0.',
             '- **Store convention:** KR input `?` predicts the remembered value, as cheap checks. MQAR is the **value-prediction write**, canonical `source−1`: the second key character is input and its value is target. Not the next input-value character, which predicts another key. There are exactly K table stores per MQAR episode.',
             '- **M2:** MQAR all unordered pairs of different table stores within each episode; KR its single store versus **every other real write step in the episode**, including later steps. Thus KR overlap is a store/other surrogate, **not literal two-item store/store separation**. Mean plus median/IQR; compare the mandated random baseline k/(2N−k).',
             '- **M3:** all store top-k memberships across episodes: fraction of neurons ever selected, and share of selections assigned to the most-selected **ceil(.05N)** neurons. M3 never pools neuron IDs across training seeds; KR tables show the range across three independent checkpoints.',
             '- **Actual width:** trunk N=**1033** in KR, **1070** in MQAR (concatenated input+hidden), i2h N=**1024**. Thus top64 is 6.20%/5.98%/6.25%, respectively; k64 baselines are .03197/.03083/.03226, not all .03226. No layer-width renormalization changes the thresholds.',
             '', '### Decision rule (verbatim)','']
    protocol=Path('/home/jaden/memory_encoding/scratch/p6_look_2026-11-05/PROTOCOL.md').read_text()
    text += [protocol[protocol.index('At k = 64'):protocol.index('\n\nProvenance correction')],
             '', '### Apply at k=64','',
             'Values below are pooled step/store/pair distributions for the three healthy KR seeds; per-seed verdicts are retained in the next table. MQAR has one training seed. “Act” means activation-selected **energy** share. “Worth” requires **both** inequalities; low concentration alone does not pass.',
             '', '| task / arm | layer | energy store share median | mean J / baseline | energy verdict | Act share median | Act mean J | Act verdict |',
             '|---|---|---:|---:|---|---:|---:|---|']
    activation_wins=[]
    for label,g in groups.items():
        ls=('L1','L2') if g['task']=='kr' else ('L0','L1','L2')
        for l in ls:
            e,a=[g['layers'][l]['selections'][v]['64'] for v in ('energy','activation')]
            text.append(f"| {label} | {l} | {e['store']['median']:.3f} | {e['jaccard']['mean']:.3f} / {e['baseline']:.5f} | {decision(e)} | {a['store']['median']:.3f} | {a['jaccard']['mean']:.3f} | {decision(a)} |")
        for r in g['runs']:
            for l in ls:
                e,a=[r['layers'][l]['selections'][v]['64'] for v in ('energy','activation')]
                if decision(a)=='Worth running P6' and decision(e)!='Worth running P6':
                    activation_wins.append(f"{label} seed{r['config']['seed']} {l}: energy {e['store']['median']:.3f}/J{e['jaccard']['mean']:.3f}, activation {a['store']['median']:.3f}/J{a['jaccard']['mean']:.3f}")
    text += ['', '| KR arm | seed | L1 energy share / J / verdict | L2 energy share / J / verdict |', '|---|---:|---|---|']
    for label,g in groups.items():
        if g['task']!='kr': continue
        for r in g['runs']:
            cells=[]
            for l in ('L1','L2'):
                q=r['layers'][l]['selections']['energy']['64']
                cells.append(f"{q['store']['median']:.3f} / {q['jaccard']['mean']:.3f} / {decision(q)}")
            text.append(f"| {label} | {r['config']['seed']} | "+' | '.join(cells)+' |')
    text += ['', '**Activation passes where energy does not:** '+('; '.join(activation_wins) if activation_wins else 'none in this panel.')+'.',
             '', '**M3 at k64, memory layers:** energy selection uses 30–44% of KR neurons at least once across 128 episodes, with the most-used 5% taking 26–31% of all selections (ranges across layers/seeds). MQAR L0 uses about 86% of neurons, with its top 5% taking 16–17% of selections at every K. Thus the bigkey failure of the fixed gate is **concentration**, not the same small set taking every item; its mean within-episode J is only about 1.8–1.9× random. Activation selection is more skewed in MQAR L0 (top5 share 37–56%) and does not clear the overlap gate.', '', '## Which MQAR layer carries recall?','',
             'Strict read-time knockout on the same test batches subtracts just one layer’s fast read **at recall prediction steps**, leaving store-time computation unchanged. The first-answer column cannot be contaminated by earlier answer writes. **L0 is the net recall-supporting fast layer at every K:** removing its fast read reduces first-answer accuracy from .781/.672/.461/.273 to .398/.313/.195/.133 (K4/8/16/32). In contrast, removing L1 or L2 improves recall: their fast reads are net harmful at query time in these pilots. Apply the task decision to L0 (**Not worth it** at every K, because median top64 share > .8); the L1/L2 ambiguous rows are not the bigkey task verdict. This diagnostic localizes dependence; it is not a new concentration decision metric. Report all trunk layers rather than selecting whichever gives a favorable gate. i2h writes are shown in the full tables but **cannot carry recall with recurrence off**.',
             '', '| K | condition | strict first answer | strict all targets |', '|---:|---|---:|---:|']
    for k,p in probes.items():
        for name,q in p.items():
            text.append(f"| {k} | {'baseline' if name=='none' else 'remove '+name+' fast read at recall'} | {q['first']:.3f} | {q['recall']:.3f} |")
    text += ['', '## M1–M3 full layer tables','',
             'All fractions are 0–1. M1 cells are **median [Q25–Q75]**. M2 cells are **mean; median [Q25–Q75]**. KR display distributions pool three seeds at the step/pair level, not neuron identities. M3 cells are **ever selected / top-5%-selection share**, ranges across seeds for KR and single values for MQAR. Exact per-checkpoint JSON and raw arrays retain all counts and distributions.',
             '', '### Participation ratio (same for both selectors)','',
             '| task / arm | layer | N | real steps | zero-energy | store steps | PR overall | PR store |','|---|---|---:|---:|---:|---:|---|---|']
    for label,g in groups.items():
        for l,d in g['layers'].items():
            text.append(f"| {label} | {l} | {d['N']} | {d['real_steps']} | {d['zero_energy_steps']} | {d['store_steps']} | {dist(d['pr_all'])} | {dist(d['pr_store'])} |")
    for label,g in groups.items():
        text += ['',f'### {label} — M1 energy retained','',
                 '| layer | selection | k16 overall / store | k64 overall / store | k256 overall / store |',
                 '|---|---|---|---|---|']
        for l,d in g['layers'].items():
            for v in ('energy','activation'):
                cells=[dist(d['selections'][v][k]['all'])+' / '+dist(d['selections'][v][k]['store']) for k in ('16','64','256')]
                text.append(f'| {l} | {v} | '+' | '.join(cells)+' |')
        text += ['',f'### {label} — M2 overlap and M3 reuse','',
                 '| layer | selection | k | random J | M2 mean; median [IQR] | M3 ever / top5 share |',
                 '|---|---|---:|---:|---|---|']
        for l,d in g['layers'].items():
            for v in ('energy','activation'):
                for k in ('16','64','256'):
                    q=d['selections'][v][k]
                    ever=q.get('ever_selected_by_seed',[q.get('ever_selected')])
                    top=q.get('top5_share_by_seed',[q.get('top5_share')])
                    text.append(f"| {l} | {v} | {k} | {q['baseline']:.5f} | {q['jaccard']['mean']:.3f}; {dist(q['jaccard'])} | {rng(ever)} / {rng(top)} |")
    sensitivity=read('kr_plain_ln_s3141')
    text += ['', '## Caveats and reproducibility','',
             '- This is an **analysis of dense, existing writes**, not a causal test of a retrained sparse writer or its energy-matching scale. Low top-k share does not imply retaining only that k can preserve recall. Activation selection may discard most energy even when overlap is low.',
             '- KR has only one stored item per sequence. Its rule uses the requested store-vs-every-other comparison; it cannot establish two-item slot separation. KR L0 is concentrated because one-hot keys and exclusive masks restrict eligible rows, but the designated memory layers are L1/L2.',
             '- **128 episodes/checkpoint**, only one MQAR training seed, and first stored batches only; no confidence intervals or seed-replication claim for MQAR. Pair observations within an episode are not independent. K-specific checkpoints differ through training as well as load.',
             '- Reuse depends on this finite population: “ever selected” is not an asymptotic utilization estimate. Stable tie-breaking is explicit; zero-energy M1 exclusions are shown in the PR table, not silently assigned a share.',
             '- Initial plain-ln s3141 sensitivity (excluded for prior strict health): '+ '; '.join(f"{l} share {sensitivity['layers'][l]['selections']['energy']['64']['store']['median']:.3f}, J {sensitivity['layers'][l]['selections']['energy']['64']['jaccard']['mean']:.3f}, {decision(sensitivity['layers'][l]['selections']['energy']['64'])}" for l in ('L1','L2'))+'. The exclusion does not turn an unfavorable energy verdict into a favorable one.',
             '- Code and fixed protocol: `manticore:/home/jaden/memory_encoding/scratch/p6_look_2026-11-05/`, branch **p6-look** (separate worktree `~/p6-look`), no main merge and no library changes. Revival audit (`revival_verification.json`) confirmed all six replay/import files in each archive byte-identical to their named git commit and all four copied checkpoint hashes unchanged. `test_measure.py`: 3 CPU tests pass (exact row-energy algebra; uniform/disjoint/tie checks; padding/zero handling). Actual checkpoint loads and completed replays serve as integration checks.',
             '- Full results: `manticore:/home/jaden/p6_look_2026-11-05/results/` (`*.json`, per-layer `*.npz`); `pooled_summary.json`, `analysis.log`, hash manifests. `run.sh`, `probe_layers.py`, `report.py` reproduce the analysis; no figure adds information beyond these fixed-k tables, so none included.',
             '', '## Five-line summary','']
    text += ['1. Existing checkpoints only: nine healthy KR runs and four MQAR lin pilots; no training or main merge.',
             '2. KR is ambiguous in every primary arm and memory layer: top64 share ≈0.32–0.33, but J≈0.133–0.140 misses the fixed overlap gate (0.09590).',
             '3. MQAR K4/8/16/32: not worth it in recall-supporting L0; top64 already carries 0.803–0.806 of write energy (J≈0.055–0.058).',
             '4. Activation does not pass in any pooled task/arm memory layer; DC2-ln seed2718 L2 is the lone individual-seed variant-only pass.',
             '5. This is a screening look, not evidence that sparse writes improve recall; Jaden decides whether P6 runs.']
    NOTE.write_text('\n'.join(text)+'\n')
    print(NOTE)
    print('\n'.join(verdicts))
    print('activation-only passes',activation_wins)

if __name__=='__main__':
    main()
