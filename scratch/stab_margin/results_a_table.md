# Part (a): stability-margin proxy analysis

## Short README

Values are read directly from the supplied JSON files; no checkpoints are interpolated. SV2/SV1 are the leading slow singular values for trunk 2/trunk 1. `s0.act_med` is the final slow-only trunk activation median and `s0.gact_med` is its pre-LN/pre-tanh GELU norm. The latter is the comparable growth measure for ST1 (LayerNorm hides growth in `act_med`) and ST4 (output tanh can hide saturation). Lineage feature files contain no replay activation or loss fields, so those cells are shown as —. ST2's trend summary ends at 290k, before the reported 295.5k collapse; its measured 300k row is still displayed.

## Summary statistics

Spearman trends use every available measured checkpoint from 100k through the stated endpoint. Ratios use the exact endpoint pair shown.

| Arm | Ratio endpoints | SV2 ratio | s0.gact ratio | Spearman(iter, SV2) | Spearman(iter, gact) |
| --- | --- | --- | --- | --- | --- |
| ST1 (LayerNorm) | 600k/150k | 1.908 | 1.791 | 1 (n=51) | 1 (n=51) |
| ST2 (clamp 0.3) | 290k/150k | 1.798 | 2.653 | 1 (n=20) | 0.9985 (n=20) |
| ST3 (grad clip 30) | 600k/150k | 1.441 | 2.596 | 1 (n=51) | 0.9525 (n=51) |
| ST4 (output tanh) | 600k/150k | 4.096 | 29.46 | 1 (n=51) | 0.989 (n=51) |
| B (plain, seed 3141) | 205k/150k | 1.341 | 2.092 | 1 (n=15) | 0.8714 (n=15) |
| plain 4241 | 150k/50k | 1.825 | — | 1 (n=2) | — (n=0) |
| plain 2718 | 150k/50k | 2.179 | — | 1 (n=2) | — (n=0) |

## Naive plain-recipe edge prediction

Fit on 10 plain-recipe health rows at stage ≥40k: `log(edge) = 0.8188 + (-0.3543) log(SV2)` (log-space R²=0.762). The observed fitting range is SV2 6.630–20.995. This is only a naive proxy transfer: predictions outside that interval are explicitly marked **EXTRAPOLATION** and should not be interpreted as calibrated stability margins.

| Arm | Stage | SV2 | Naive predicted edge | Range flag |
| --- | --- | --- | --- | --- |
| ST1 (LayerNorm) | 150k | 47.43 | 0.5779 | **EXTRAPOLATION** |
| ST1 (LayerNorm) | 300k | 69.54 | 0.5047 | **EXTRAPOLATION** |
| ST1 (LayerNorm) | 450k | 81.18 | 0.4777 | **EXTRAPOLATION** |
| ST1 (LayerNorm) | 600k | 90.51 | 0.4597 | **EXTRAPOLATION** |
| ST2 (clamp 0.3) | 150k | 8.376 | 1.068 | within fit SV2 range |
| ST2 (clamp 0.3) | 300k | 16.83 | 0.8343 | within fit SV2 range |
| ST2 (clamp 0.3) | 450k | — | — | missing checkpoint |
| ST2 (clamp 0.3) | 600k | — | — | missing checkpoint |
| ST3 (grad clip 30) | 150k | 8.482 | 1.063 | within fit SV2 range |
| ST3 (grad clip 30) | 300k | 11.09 | 0.9669 | within fit SV2 range |
| ST3 (grad clip 30) | 450k | 11.94 | 0.942 | within fit SV2 range |
| ST3 (grad clip 30) | 600k | 12.22 | 0.9344 | within fit SV2 range |
| ST4 (output tanh) | 150k | 77.13 | 0.4865 | **EXTRAPOLATION** |
| ST4 (output tanh) | 300k | 200 | 0.3471 | **EXTRAPOLATION** |
| ST4 (output tanh) | 450k | 287.6 | 0.3052 | **EXTRAPOLATION** |
| ST4 (output tanh) | 600k | 315.9 | 0.2952 | **EXTRAPOLATION** |

## SV2 sanity check at 100k

A strong same-iteration difference is defined here as at least 2× B or at most 0.5× B.

| Arm | Arm SV2 | B SV2 | Fold vs B | Strong? |
| --- | --- | --- | --- | --- |
| ST1 (LayerNorm) | 35.08 | 13.48 | 2.603 | yes |
| ST2 (clamp 0.3) | 7.34 | 13.48 | 0.5446 | no |
| ST3 (grad clip 30) | 7.284 | 13.48 | 0.5405 | no |
| ST4 (output tanh) | 43.94 | 13.48 | 3.26 | yes |

## Requested stage values

Only available requested-stage checkpoints are listed. For B, SV2 is specifically read from `replay_B.json` as `svals[2][0]`, as requested.

| Arm | Stage | SV2 | SV1 | s0.act_med | s0.gact_med | s0 loss | s1 loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ST1 (LayerNorm) | 50k | 36.72 | 61.27 | 32.14 | 551.5 | 2.311 | 0.7331 |
| ST1 (LayerNorm) | 100k | 35.08 | 81.61 | 32.14 | 516 | 2.168 | 0.7105 |
| ST1 (LayerNorm) | 150k | 47.43 | 93.68 | 32.14 | 619 | 2.281 | 0.6622 |
| ST1 (LayerNorm) | 200k | 56.7 | 101.3 | 32.14 | 696 | 2.268 | 0.6819 |
| ST1 (LayerNorm) | 250k | 64.05 | 106.1 | 32.14 | 779.8 | 2.402 | 0.649 |
| ST1 (LayerNorm) | 300k | 69.54 | 109.4 | 32.14 | 858.3 | 2.482 | 0.6424 |
| ST1 (LayerNorm) | 350k | 73.81 | 112.1 | 32.14 | 927.2 | 2.506 | 0.6242 |
| ST1 (LayerNorm) | 400k | 77.69 | 114.3 | 32.14 | 980 | 2.517 | 0.6113 |
| ST1 (LayerNorm) | 450k | 81.18 | 116.3 | 32.14 | 1015 | 2.48 | 0.6099 |
| ST1 (LayerNorm) | 500k | 84.61 | 118.3 | 32.14 | 1041 | 2.495 | 0.5858 |
| ST1 (LayerNorm) | 550k | 87.67 | 120.1 | 32.14 | 1063 | 2.523 | 0.632 |
| ST1 (LayerNorm) | 600k | 90.51 | 121.8 | 32.14 | 1109 | 2.533 | 0.634 |
| ST2 (clamp 0.3) | 50k | 5.733 | 4.749 | 14.11 | 14.11 | 3.729 | 0.9139 |
| ST2 (clamp 0.3) | 100k | 7.34 | 5.307 | 18.52 | 18.52 | 3.579 | 0.8816 |
| ST2 (clamp 0.3) | 150k | 8.376 | 5.751 | 26.69 | 26.69 | 2.811 | 2.232 |
| ST2 (clamp 0.3) | 200k | 10.1 | 6.27 | 32.04 | 32.04 | 2.221 | 1.178 |
| ST2 (clamp 0.3) | 250k | 12.27 | 7.194 | 45.08 | 45.08 | 2.131 | 0.7666 |
| ST2 (clamp 0.3) | 300k | 16.83 | 8.938 | 85.67 | 85.67 | 5.111 | 3.321 |
| ST3 (grad clip 30) | 50k | 7.012 | 5.195 | 22.5 | 22.5 | 3.457 | 0.9953 |
| ST3 (grad clip 30) | 100k | 7.284 | 5.48 | 22.84 | 22.84 | 2.9 | 0.9421 |
| ST3 (grad clip 30) | 150k | 8.482 | 6.329 | 23.81 | 23.81 | 2.04 | 1.033 |
| ST3 (grad clip 30) | 200k | 9.429 | 6.841 | 24.76 | 24.76 | 2.098 | 0.871 |
| ST3 (grad clip 30) | 250k | 10.41 | 7.24 | 37.13 | 37.13 | 2.439 | 0.7743 |
| ST3 (grad clip 30) | 300k | 11.09 | 7.76 | 48.98 | 48.98 | 2.525 | 0.7191 |
| ST3 (grad clip 30) | 350k | 11.53 | 8.103 | 56.46 | 56.46 | 2.365 | 0.6958 |
| ST3 (grad clip 30) | 400k | 11.85 | 8.385 | 60.25 | 60.25 | 2.16 | 0.6759 |
| ST3 (grad clip 30) | 450k | 11.94 | 8.54 | 61.69 | 61.69 | 2.184 | 0.7136 |
| ST3 (grad clip 30) | 500k | 12.01 | 8.615 | 62 | 62 | 2.11 | 0.6392 |
| ST3 (grad clip 30) | 550k | 12.12 | 8.663 | 61.85 | 61.85 | 2.298 | 0.7413 |
| ST3 (grad clip 30) | 600k | 12.22 | 8.709 | 61.81 | 61.81 | 2.204 | 0.6036 |
| ST4 (output tanh) | 50k | 22.6 | 6.938 | 61.19 | 61.19 | 1.966 | 0.882 |
| ST4 (output tanh) | 100k | 43.94 | 10.83 | 147.6 | 147.6 | 2.137 | 0.8591 |
| ST4 (output tanh) | 150k | 77.13 | 14.27 | 352.2 | 352.2 | 2.085 | 0.9283 |
| ST4 (output tanh) | 200k | 113.7 | 17.45 | 1135 | 1135 | 1.994 | 0.9341 |
| ST4 (output tanh) | 250k | 152.2 | 20.2 | 2253 | 2253 | 2.059 | 0.9535 |
| ST4 (output tanh) | 300k | 200 | 23.79 | 3471 | 3471 | 2.09 | 0.8556 |
| ST4 (output tanh) | 350k | 246.7 | 26.59 | 6036 | 6036 | 2.136 | 0.8741 |
| ST4 (output tanh) | 400k | 272 | 27.92 | 7433 | 7433 | 2.067 | 0.9496 |
| ST4 (output tanh) | 450k | 287.6 | 29.37 | 8010 | 8010 | 1.925 | 0.9105 |
| ST4 (output tanh) | 500k | 298.1 | 30.39 | 8810 | 8810 | 2.017 | 0.8846 |
| ST4 (output tanh) | 550k | 308.3 | 31.91 | 9139 | 9139 | 2.063 | 0.8366 |
| ST4 (output tanh) | 600k | 315.9 | 32.89 | 1.038e+04 | 1.038e+04 | 2.062 | 0.8294 |
| B (plain, seed 3141) | 100k | 13.48 | 8.054 | 24.34 | 24.34 | 2.095 | 1.241 |
| B (plain, seed 3141) | 150k | 16.51 | 9.46 | 24.52 | 24.52 | 2.058 | 1.148 |
| B (plain, seed 3141) | 200k | 21.33 | 11.63 | 48.48 | 48.48 | 3.24 | 5.553 |
| plain 4241 | 50k | 6.63 | 6.771 | — | — | — | — |
| plain 4241 | 100k | 10.4 | 9.083 | — | — | — | — |
| plain 4241 | 150k | 12.1 | 10.84 | — | — | — | — |
| plain 2718 | 50k | 9.634 | 6.916 | — | — | — | — |
| plain 2718 | 100k | 15.02 | 8.624 | — | — | — | — |
| plain 2718 | 150k | 21 | 10.99 | — | — | — | — |

## Data sanity

The script recursively checks all supplied inputs and prints every NaN/inf path. Nonfinite values occur in replay diagnostic fields such as gain/frac values when fast writes are off; requested extracted measurements and generated JSON are finite or null. Counts by input:

| Input | NaN/inf count |
| --- | --- |
| replay_ST1.json | 180 |
| replay_ST2.json | 90 |
| replay_ST3.json | 180 |
| replay_ST4.json | 180 |
| replay_B.json | 57 |
| feat_local.json | 0 |
| feat_L.json | 0 |
| health_rows.json | 0 |

Figure: [fig_a_proxy.png](fig_a_proxy.png). Machine-readable output: [results_a.json](results_a.json).
