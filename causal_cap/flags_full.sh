# Arms of the full-spectrum norm-cap follow-up (2026-10-02 evening). source flags.sh first (CTRL recipe).
COMMON_FULL=(--n_iters 300000 --print_freq 500 --trace_loop_every 500 --checkpoint_save_freq 5000
  --checkpoint_keep_every 10000 --checkpoint_keep_max 100 --early_stop_window 0)
arm_flags_full() {  # NAME SPECDIR
  case $1 in
    CAPALL_T)   ARM_FLAGS=(--norm_cap_file $2/capall_t.pt) ;;
    CAPALL_TRB) ARM_FLAGS=(--norm_cap_file $2/capall_trb.pt) ;;
    FROB)       ARM_FLAGS=(--norm_cap_file $2/frob.pt) ;;
    CAPALL_TRB_late) ARM_FLAGS=(--norm_cap_file $2/capall_trb.pt --norm_cap_start 170000) ;;  # START_CKPT = CTRL 170k
    *) return 1 ;;
  esac
}
