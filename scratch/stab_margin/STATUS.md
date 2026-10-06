# STATUS (stab-margin)
2026-10-06: started. Worktree ~/memory_encoding/.claude/worktrees/stab-margin (branch stab-margin).
Deckard ST1/3/4 ckpts 10k..600k every 10k (312MB each), ST2 to 300k. Deckard GPUs free (2xA6000).
Step 1: Codex recon of two-modes tooling (running).
Prereg written (vault note, section 1) 2026-10-06 before any cell. Cells run on DECKARD GPUs (A6000 ~80 it/s, 50k cell ~11 min), not ORC.
Part (a) ext replays 310k..600k running on deckard analysis_st/ext_*.log
Phase1 cells launched on deckard 2 lanes (cells_p1_g0/g1.txt). part (a) Codex analysis running
Part (a) done (scratch/stab_margin/results_a_table.md): SV2 ratios 600k/150k ST1 1.91, ST3 1.44 (plateau ~12), ST4 4.10, ST2 1.80 to 290k. Waiting on cells + collector (Codex).
Phase2 (300k,450k x m1,1.5,2,3, 3 arms) chained after phase1 on deckard (cells_p2_g*.txt). Cells ~13 min each.
Chains for phase 2 cancelled; phase 2 to be redesigned after phase 1 results (ST1 edge >3 at 150k/600k; ST3 150k edge ~3 [T at m=3]).
NOTE 2026-10-06: Codex short-window limit resets 3:20pm MDT. Waiting on Codex (if limited): final note drafting/figures refresh. Cells keep running on deckard.
Phase2 = m 5,10 at 150k/600k for ST1/3/4 (12 cells), chained. Phase1 result: all S for m<=3 except ST3_150k_m3 = T.
Phase3 launched 14:35: ST1/ST4 m=30,100 at 150k/600k; ST3 reseed 11 at boundary (12 cells, ~1.5h). After this total ~12 GPU-h = cap. Phase2 result: ST1/ST4 stable to m=10 at 150k,600k; ST3 edge 3 (150k) -> 5 (600k).
