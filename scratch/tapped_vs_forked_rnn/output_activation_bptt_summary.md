# Forked output-activation ablation (standalone BPTT)

## Design

- 20 paired seeds per arm and task; 120 runs total on CPU with 12 workers.
- Recurrence fixed to `h_next = tanh(W_state(z))`; topology fixed to forked.
- The only varied operation was the parameter-free transform before the shared-shaped output layer: identity, `tanh`, or ReLU.
- All arms used the same 5,884-parameter module graph, seed-matched initialization and batches, Adam (`lr=0.003`), full BPTT, gradient clipping at 1.0, curriculum, and budgets (1,800 copy / 2,600 palindrome steps).
- Evaluations include in-distribution, longer lengths, palindrome fixed-3/fixed-4, copy 8--10 stress, and palindrome fixed-6 stress. CIs are two-sided 95% Student-t intervals over paired seed differences.

## Identity minus tanh effects

| Task | Panel | Token accuracy difference (95% CI) | Exact accuracy difference (95% CI) |
|---|---|---:|---:|
| Copy | in-distribution | +0.024866 [+0.009564, +0.040168] | +0.050977 [+0.016039, +0.085914] |
| Copy | longer 5--6 | +0.022429 [+0.000645, +0.044213] | +0.010742 [+0.002378, +0.019107] |
| Copy | stress 8--10 | +0.001029 [-0.014822, +0.016879] | +0.000000 [-0.000514, +0.000514] |
| Two-palindrome | in-distribution | +0.009943 [-0.011447, +0.031334] | +0.036719 [-0.018952, +0.092390] |
| Two-palindrome | longer 4--5 | +0.008722 [-0.003903, +0.021347] | +0.000000 [-0.002055, +0.002055] |
| Two-palindrome | fixed-3 | +0.010474 [-0.010385, +0.031333] | +0.015137 [-0.008854, +0.039127] |
| Two-palindrome | fixed-4 | +0.008191 [-0.006243, +0.022625] | +0.000781 [-0.001811, +0.003373] |
| Two-palindrome | fixed-6 stress | +0.003076 [-0.005373, +0.011525] | +0.000000 [0.000000, 0.000000] |

Identity clearly improved copy in-distribution and modestly improved copy length-5--6 generalization. The corresponding palindrome point estimates favored identity, but every CI crossed zero. Neither transform solved far extrapolation: copy 8--10 token accuracy was 0.3757 (identity), 0.3747 (tanh), and 0.3801 (ReLU), while fixed-6 palindrome exact accuracy was zero in every run and arm.

## Stability and diagnostics

- All 120 runs completed; there were zero training failures, nonfinite failures, or nonfinite evaluation panels. Validation confirmed no missing/duplicate keys, equal parameter counts, byte-equal paired initializations, paired batch-stream fingerprints, and complete finite outputs.
- Mean clip-binding fractions (copy / palindrome) were identity 0.7641 / 0.9549, tanh 0.6897 / 0.8954, and ReLU 0.8193 / 0.9321. Thus output tanh reduced clipping frequency, though clipping remained common.
- In-distribution readout-feature RMS (copy / palindrome) was identity 1.1463 / 1.3062, tanh 0.6906 / 0.6902, and ReLU 1.4519 / 1.2419. Identity-minus-tanh feature-RMS differences were +0.4557 [+0.4092, +0.5023] on copy and +0.6160 [+0.5811, +0.6510] on palindrome.
- In-distribution logit RMS (copy / palindrome) was identity 4.9764 / 5.2763 versus tanh 3.6617 / 3.8534. Identity-minus-tanh differences were +1.3148 [+1.0086, +1.6209] and +1.4229 [+1.1034, +1.7424].
- Tanh readout saturation (`abs(tanh(z)) > .95`) was modest: 6.93% in-distribution and 7.60% at copy 8--10; 6.67% in-distribution and 6.52% at palindrome fixed-6.
- ReLU was a useful secondary reference but underperformed both primary arms in-distribution (copy token/exact 0.8213/0.5846; palindrome 0.7268/0.2203). Its stress copy token accuracy was statistically unremarkable despite the slightly larger point estimate.

## Execution and artifacts

Shell command:

```bash
/usr/bin/time -p python "output_activation_bptt.py" --seeds 20 --workers 12 --output output_activation_bptt_results.json > output_activation_bptt_run.log 2>&1
```

Measured experiment runtime was 561.176 s (script wall clock); `/usr/bin/time` reported 565.81 s real, 6195.19 s user, and 9.81 s sys. Outputs are `output_activation_bptt_results.json`, `output_activation_bptt_summary.csv`, `output_activation_bptt.png`, and `output_activation_bptt_run.log`.

## Caveats

- This is one compact synthetic architecture and one optimizer/curriculum/budget setting; it isolates output representation but does not establish a general architectural ranking.
- Exact accuracy is floor-limited on stress panels, so token accuracy is more informative there.
- Evaluation CIs quantify across-seed paired variation, not uncertainty over task families or hyperparameters.
- PyTorch emitted a CUDA-driver probe warning in worker processes, but the configured and recorded device was CPU and all deterministic validation checks passed.
