# Tapped vs. forked vanilla RNN experiment

This standalone PyTorch experiment compares where a shared output layer reads a
deep vanilla RNN. Every arm has the same embedding, two-layer ReLU residual trunk,
state projection, and output projection (and therefore exactly the same
parameter count). Only the tensor supplied to the output projection changes:

- `tapped_post_tanh`: `z = h_(t-1) + trunk([embedding(x_t), h_(t-1)])`;
  (the trunk is two Linear+ReLU transformations); `u = W_state(z)`;
  `h_t = tanh(u)`; output reads `h_t`.
- `tapped_pre_tanh_relu`: the same recurrence; output reads `ReLU(u)`.
- `forked`: the same recurrence; output reads `z`.

The script uses Adam, full BPTT (no detach), deterministic CPU execution, and
identical generated batch streams across arms for a given task and seed. Its
expanded-panel default is 20 seeds. Loss and accuracy are masked to recall
positions only.

Tasks:

1. **Delayed copy:** random 2–4-symbol block, delimiter, 1–3 blanks, query,
   then recall the block. Generalization uses lengths 5–6.
2. **Two-palindrome recall:** two independent random 2–3-symbol blocks are
   presented before delayed, separately queried reverse recalls.
   Generalization uses lengths 4–5 for both blocks. A deterministic length
   curriculum expands the upper bound every 200 steps; most training occurs on
   the full variable-length distribution.

Run from this directory:

```bash
python experiment.py
```

Outputs are `results.json` (complete config, per-run curves and results),
`summary.csv`, and one PNG per task. The measured results and interpretation
follow.

## Results

The exact full-run command was `python experiment.py`. It ran on CPU with
PyTorch 2.9.0, Python 3.12.9, and deterministic algorithms enabled. Total
wall-clock time was **1031.6 s (17m 11.6s)**. Every arm had **5,884 trainable
parameters** (asserted by the script).

Accuracy is mean ± sample SD across five seeds:

| Task | Readout | In-distribution | Longer lengths | Seeds reaching 90% | Mean step among successes |
|---|---|---:|---:|---:|---:|
| copy | post-tanh | 83.8% ± 5.9% | 56.1% ± 4.7% | 1/5 | 1325 (84,800 examples) |
| copy | pre-tanh ReLU | 86.8% ± 2.7% | **59.6% ± 2.2%** | 1/5 | 1250 (80,000 examples) |
| copy | forked | **88.4% ± 4.8%** | 58.5% ± 4.6% | **3/5** | 1500 (96,000 examples) |
| two-palindrome | post-tanh | 74.8% ± 1.5% | 50.0% ± 1.5% | 0/5 | — |
| two-palindrome | pre-tanh ReLU | 73.1% ± 3.8% | 48.5% ± 2.6% | 0/5 | — |
| two-palindrome | forked | **81.5% ± 4.1%** | **53.1% ± 2.0%** | 0/5 | — |

No run reached 99%. Thresholds are the first fixed-validation evaluation at or
above the target (evaluated every 25 optimizer steps), so missing values are
censored at 1,800 copy steps or 2,600 palindrome steps rather than infinite.

## Interpretation

Forking the readout from the shared trunk was strongest in-distribution on both
tasks and was notably more reliable at crossing 90% on copy. The favored
pre-tanh-ReLU tap improved mean copy accuracy and longer-length copy transfer
over the textbook post-tanh tap, with lower variance, but it did not beat the
fork on the harder two-block reversal. All methods generalized poorly relative
to their training lengths, so these results support a modest topology effect,
not robust length extrapolation. With only five paired seeds, differences
should be treated as descriptive rather than definitive.

## Expanded paired panel (20 seeds)

This section is separate from, and does not replace, the original five-seed
results above. The expanded run makes `tapped_post_tanh` versus `forked` the
primary topology comparison; `tapped_pre_tanh_relu` is retained as a secondary
diagnostic. No architecture, optimizer, training distribution, step budget, or
full-BPTT behavior changed. Each process used one PyTorch CPU thread, and every
arm received the same deterministic batches for each task/seed pair.

Exact command:

```bash
/home/jaden/miniforge3/bin/python experiment.py --seeds 20 --workers 12 --output expanded_results.json
```

The run took **373.3 s (6m 13.3s)** with 12 workers. All arms again had exactly
**5,884 parameters**. Standard evaluation panels use four deterministic batches
(256 samples) per seed. The added fixed-length two-palindrome panels use 16
deterministic batches (1,024 samples) per seed at block length 3 and separately
at block length 4. “Exact” requires every recalled target token across both
reversed blocks (or the whole copy block) to be correct. Values below are mean ±
sample SD over paired seeds.

| Task / panel | Readout | Token accuracy | Exact-sequence accuracy |
|---|---|---:|---:|
| copy, train lengths 2–4 | post-tanh | 87.18% ± 3.81% | 68.18% ± 7.41% |
|  | pre-tanh ReLU | 86.97% ± 2.36% | 67.91% ± 5.16% |
|  | forked | **89.77% ± 3.17%** | **73.50% ± 7.35%** |
| copy, longer 5–6 | post-tanh | 58.14% ± 3.37% | 3.95% ± 1.45% |
|  | pre-tanh ReLU | 58.51% ± 2.32% | 3.75% ± 1.34% |
|  | forked | **59.37% ± 3.15%** | **4.98% ± 1.67%** |
| two-palindrome, train lengths 2–3 | post-tanh | 75.93% ± 4.06% | 23.98% ± 10.09% |
|  | pre-tanh ReLU | 74.62% ± 3.77% | 23.46% ± 8.01% |
|  | forked | **80.54% ± 4.30%** | **36.95% ± 11.49%** |
| two-palindrome, longer 4–5 | post-tanh | 50.21% ± 2.02% | 0.20% ± 0.30% |
|  | pre-tanh ReLU | 48.57% ± 2.33% | 0.16% ± 0.20% |
|  | forked | **52.29% ± 1.55%** | **0.37% ± 0.50%** |
| two-palindrome, fixed 3 | post-tanh | 69.12% ± 3.87% | 8.47% ± 4.68% |
|  | pre-tanh ReLU | 67.99% ± 3.63% | 7.76% ± 3.48% |
|  | forked | **73.81% ± 3.92%** | **15.12% ± 5.98%** |
| two-palindrome, fixed 4 | post-tanh | 55.12% ± 2.30% | 0.48% ± 0.33% |
|  | pre-tanh ReLU | 53.73% ± 2.07% | 0.36% ± 0.14% |
|  | forked | **57.53% ± 2.00%** | **0.82% ± 0.39%** |

For the primary paired contrast (forked minus post-tanh), two-sided 95% Student
*t* confidence intervals and fork/tie/post win counts were:

| Panel | Token difference (95% CI), wins | Exact difference (95% CI), wins |
|---|---:|---:|
| copy, train lengths | +2.59 pp [0.58, 4.61], 12/2/6 | +5.31 pp [1.28, 9.34], 13/0/7 |
| copy, longer | +1.23 pp [-0.97, 3.43], 12/0/8 | +1.04 pp [0.06, 2.01], 13/2/5 |
| palindrome, train lengths | +4.61 pp [2.60, 6.62], 17/1/2 | +12.97 pp [7.29, 18.65], 17/0/3 |
| palindrome, longer | +2.07 pp [1.00, 3.15], 17/1/2 | +0.18 pp [-0.06, 0.42], 8/8/4 |
| palindrome, fixed 3 | +4.69 pp [2.78, 6.60], 17/0/3 | +6.66 pp [3.71, 9.60], 16/0/4 |
| palindrome, fixed 4 | +2.41 pp [1.15, 3.68], 15/0/5 | +0.34 pp [0.17, 0.52], 18/0/2 |

The 90%-token threshold was reached by 15/20 forked, 7/20 post-tanh, and 3/20
pre-tanh-ReLU copy runs. On two-palindrome it was reached by 1/20 forked runs
and no tapped runs. No run reached 99%. Threshold misses remain right-censored
at the unchanged training budgets.

Expanded artifacts are `expanded_results.json` (config, metadata, every curve,
per-seed evaluations, panel summaries, paired differences/CIs/wins) and
`expanded_summary.csv`; expanded plots use the `expanded_` prefix. A caveat is
that fixed-length-4 exact recall remains near zero for every arm, producing a
small absolute advantage despite a positive paired interval. The four-batch
general evaluations also have more per-seed sampling noise than the 16-batch
fixed panels.

## Recurrent-activation ablation (locked forked readout)

This is a separate 20-seed ablation in `activation_ablation.py`; it does not
alter the topology experiment above. The readout is locked to
`logits = W_out(z)` in every arm. Only the parameter-free map that forms the
next recurrent carrier from `u = W_state(z)` changes: `tanh(u)`, `relu(u)`,
`u`, or `softsign(u)`. Thus, the output never reads `u` or `h_next`. All four
arms instantiate the same modules and **5,884 parameters**, start from paired
initializations, and consume identical task/seed batch streams.

Exact command:

```bash
/home/jaden/miniforge3/bin/python activation_ablation.py --seeds 20 --workers 12 --output activation_ablation_results.json
```

The script-reported wall time was **541.5 s (9m 1.5s)**; `/usr/bin/time`
reported **544.60 s**. Validation passed for all **160 expected runs** (20
seeds x 4 activations x 2 tasks), with no duplicates, missing pairs,
parameter-count mismatches, worker failures, or nonfinite failures. Values
below are means across 20 paired seeds (token / exact-sequence accuracy).

| Task / evaluation panel | tanh | ReLU | identity | softsign |
|---|---:|---:|---:|---:|
| copy, train lengths 2–4 | 89.77 / 73.50% | 93.72 / 83.61% | **95.23 / 87.27%** | 84.16 / 63.38% |
| copy, longer 5–6 | 59.37 / 4.98% | **63.66** / 7.27% | 61.47 / **8.85%** | 55.98 / 3.52% |
| palindrome, train lengths 2–3 | 80.54 / 36.95% | 85.33 / 44.06% | **97.27 / 87.79%** | 75.34 / 25.66% |
| palindrome, longer 4–5 | **52.29** / 0.37% | 48.18 / 0.10% | 52.25 / **0.63%** | 48.66 / 0.20% |
| palindrome, fixed block 3 | 73.81 / 15.12% | 80.63 / 23.12% | **95.00 / 74.21%** | 68.37 / 8.41% |
| palindrome, fixed block 4 | 57.53 / 0.82% | 57.28 / 0.65% | **63.85 / 2.03%** | 53.66 / 0.47% |

The clearest paired result is identity versus tanh. Its mean token/exact
differences in percentage points, with two-sided 95% Student-t intervals, are:

| Panel | Token difference | Exact difference |
|---|---:|---:|
| copy, train lengths | +5.46 [3.87, 7.05] | +13.77 [9.99, 17.55] |
| copy, longer | +2.10 [-0.05, 4.24] | +3.87 [2.17, 5.56] |
| palindrome, train lengths | +16.74 [14.82, 18.66] | +50.84 [45.50, 56.18] |
| palindrome, longer | -0.04 [-1.36, 1.28] | +0.25 [-0.06, 0.57] |
| palindrome, fixed 3 | +21.19 [19.37, 23.02] | +59.09 [54.06, 64.11] |
| palindrome, fixed 4 | +6.31 [4.78, 7.85] | +1.21 [0.73, 1.69] |

ReLU also improved copy versus tanh (train token +3.95 pp [2.24, 5.66];
long token +4.29 pp [2.34, 6.24]) and palindrome fixed-3 token accuracy
(+6.83 pp [4.27, 9.38]), but hurt longer-palindrome token accuracy (-4.10 pp
[-6.08, -2.13]). Softsign was worse than tanh in-distribution on both tasks
(copy -5.61 pp [-8.51, -2.72]; palindrome -5.20 pp [-8.27, -2.12]). Full
paired intervals, per-seed differences, and win/tie/loss counts for every
metric and panel are in the JSON and CSV artifacts.

Threshold behavior agreed with the final scores. For copy, 90%-token success
was tanh 15/20, ReLU 20/20, identity 20/20, and softsign 4/20; no copy run
reached 99%. For two-palindrome it was 1/20, 6/20, 20/20, and 0/20 at 90%,
respectively; identity alone reached 99% (4/20). Among the 15 copy seeds where
both identity and tanh reached 90%, identity was earlier by 341.7 steps on
average (identity-minus-tanh -341.7, 95% CI [-474.3, -209.0]). Misses are
right-censored, so the JSON also records paired success/discordance counts
rather than assigning artificial crossing times.

Stability instrumentation reveals the tradeoff. On in-distribution copy,
mean recurrent-state RMS / mean per-run maximum absolute state was 0.87 / 1.0
for tanh, 6.32 / 86.2 for ReLU, 4.49 / 49.9 for identity, and 0.65 / 0.95 for
softsign. On in-distribution palindrome these were 0.86 / 1.0, 5.65 / 76.3,
3.83 / 37.2, and 0.60 / 0.95. Mean logit RMS was respectively 4.98, 9.69,
7.34, and 5.50 on copy and 5.28, 8.68, 9.08, and 5.13 on palindrome. No arm
became nonfinite, but the unbounded carriers relied heavily on clipping:
mean fractions of optimizer steps binding the 1.0 global clip were 76.4%,
89.9%, 89.8%, and 62.3% on copy, and 95.5%, 99.8%, 99.7%, and 80.1% on
palindrome (tanh, ReLU, identity, softsign). Mean per-run pre-clip gradient
norms were 2.16/13.01/12.22/1.83 on copy and 2.83/7.56/5.13/2.18 on
palindrome; the corresponding mean per-run maxima reached 58.5/4766/7407/75.5
and 46.7/410/421/53.0.

The main caveats are that identity's large improvement is accompanied by much
larger hidden/logit magnitudes and near-continuous clipping, length
extrapolation remains weak (especially exact two-palindrome recall), general
panels use only four batches per seed, and the result is specific to this
identity-centered initialization, short synthetic sequence regime, optimizer,
and fixed clip. Artifacts are `activation_ablation_results.json`,
`activation_ablation_summary.csv`, and the two `activation_ablation_*.png`
plots.

## Activation robustness: clipping x recurrent initialization

This targeted follow-up is separate from the activation ablation above. It keeps
the topology strictly forked (`logits = W_out(z)`), uses full BPTT and Adam, and
retains the same modules, dimensions, tasks, curricula, training distributions,
and 1,800/2,600-step budgets. It crosses `tanh` versus `identity`, global norm
clipping at 1.0 versus no clipping, and exact-identity versus orthogonal
`W_state` initialization. The orthogonal initialization uses gain **1.0**, the
neutral standard choice that preserves Euclidean norm at initialization rather
than intentionally expanding or contracting an ungated recurrent transition.
For each seed and initialization, activation and clipping cells begin with the
same parameter tensors and consume the same deterministic batches.

Exact command:

```bash
/home/jaden/miniforge3/bin/python activation_robustness.py --seeds 20 --workers 12 --output activation_robustness_results.json
```

The script-reported runtime was **1076.0 s (17m 56.0s)**; `/usr/bin/time`
reported **1079.07 s (17m 59.07s)**. Validation passed for all **320 runs** (20
seeds x 2 tasks x 2 activations x 2 clipping modes x 2 initializations), with no
duplicates, missing pairs, parameter-count mismatches, training failures, or
nonfinite evaluations. Every cell has **5,884 parameters**. ReLU was omitted
because the primary factorial panel alone took nearly 18 minutes, leaving no
comfortable margin below the requested 20-minute limit.

The central factorial simple effects below are paired identity-minus-tanh
in-distribution token-accuracy differences in percentage points (two-sided 95%
Student-*t* intervals over 20 seeds):

| `W_state` initialization | Gradient clipping | Copy | Two-palindrome |
|---|---|---:|---:|
| exact identity | norm 1.0 | **+5.46** [3.87, 7.05] | **+16.74** [14.82, 18.66] |
| exact identity | disabled | **-17.55** [-22.59, -12.50] | **+4.19** [0.47, 7.91] |
| orthogonal, gain 1.0 | norm 1.0 | **+5.33** [4.35, 6.32] | **+2.05** [1.26, 2.85] |
| orthogonal, gain 1.0 | disabled | -0.02 [-2.94, 2.90] | +3.50 [-0.44, 7.44] |

Exact-sequence effects tell the same in-distribution story. With clipping
disabled, identity-minus-tanh was -36.80 pp [-45.72, -27.87] on copy and +9.82
pp [1.32, 18.33] on palindrome under identity initialization; under orthogonal
initialization it was +0.10 pp [-6.62, 6.81] and +11.74 pp [-0.18, 23.65],
respectively. Thus identity's large original advantage **does not survive removal
of clipping in general**: it reverses sharply on copy with identity `W_state`
and becomes indistinguishable on copy with orthogonal `W_state`. It does remain
positive on the short palindrome training distribution with identity
initialization, but is uncertain under orthogonal/no-clip. Conversely, the
advantage **does survive orthogonal initialization when clipping is retained**,
although it shrinks greatly on palindrome.

Length transfer exposes a separate limitation. At fixed palindrome length 3,
identity remains better in all four cells (+3.92 to +21.27 pp token accuracy),
but at fixed length 4 it is worse for both orthogonal cells (-7.32 and -8.37 pp)
and uncertain or worse without clipping. On the stress-only evaluations,
identity loses in every cell: copy lengths 8--10 by 3.50--14.97 pp and
palindrome fixed length 6 by 6.64--9.77 pp. Exact accuracy is essentially zero
there, as expected; these panels are primarily stability diagnostics.

No condition diverged or became nonfinite. Nevertheless, identity produced
extreme finite carriers and logits. Across the stress copy panel, mean per-run
maximum absolute state/logit was **441/246 to 664/668** for identity versus
exactly **1.0** maximum state and 15--19 maximum logit for tanh. On stress
palindrome it was **41/102 to 197/163** for identity versus state 1.0 and logits
21--27 for tanh. The instability is not only a long-unroll phenomenon:
in-distribution identity mean maximum states already range from 21.5 to 49.9
(tanh: 1.0), and mean maximum logits from 28.6 to 73.0 (tanh: 15.0--27.0).
Longer unrolls amplify this bounded-versus-unbounded difference and cause the
accuracy reversal, rather than creating the stability gap from scratch.

Artifacts are `activation_robustness.py`, `activation_robustness_results.json`,
`activation_robustness_summary.csv`, `activation_robustness.png`, and
`activation_robustness_run.log`. The JSON retains every curve, failure field,
deterministic evaluation, magnitude metric, per-seed paired difference, and
validation check; the CSV contains arm summaries, panel summaries, and the
stratified factorial contrasts.
