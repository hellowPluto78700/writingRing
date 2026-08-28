#!/usr/bin/env bash
set -euo pipefail

SUBMIT_DIR="${1:-$PWD}"
REPO_ROOT="$(git -C "$SUBMIT_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

EXP305_ARRAY=$(sbatch scripts/bash_script/SNN_Bash/run_exp_3_0_5_eval_cpu_array.bash | awk '{print $4}')
EXP305_FINAL=$(sbatch --dependency=afterok:${EXP305_ARRAY} scripts/bash_script/SNN_Bash/finalize_exp_3_0_5_cpu.bash | awk '{print $4}')

EXP306_ARRAY=$(sbatch --dependency=afterok:${EXP305_FINAL} scripts/bash_script/SNN_Bash/run_exp_3_0_6_cpu_array.bash | awk '{print $4}')
EXP306_FINAL=$(sbatch --dependency=afterok:${EXP306_ARRAY} scripts/bash_script/SNN_Bash/finalize_exp_3_0_6_cpu.bash | awk '{print $4}')

EXP307_ARRAY=$(sbatch --dependency=afterok:${EXP306_FINAL} scripts/bash_script/SNN_Bash/run_exp_3_0_7_cpu_array.bash | awk '{print $4}')
EXP307_FINAL=$(sbatch --dependency=afterok:${EXP307_ARRAY} scripts/bash_script/SNN_Bash/finalize_exp_3_0_7_cpu.bash | awk '{print $4}')

printf 'Experiments 3.0.5 -> 3.0.7 pipeline submitted\n'
printf '  3.0.5 frozen probes: %s\n' "$EXP305_ARRAY"
printf '  3.0.5 finalizer:     %s\n' "$EXP305_FINAL"
printf '  3.0.6 causal heads:  %s\n' "$EXP306_ARRAY"
printf '  3.0.6 finalizer:     %s\n' "$EXP306_FINAL"
printf '  3.0.7 online eval:   %s\n' "$EXP307_ARRAY"
printf '  3.0.7 finalizer:     %s\n' "$EXP307_FINAL"
printf '\nMonitor with:\n'
printf '  squeue -u %s\n' "$USER"
printf '  sacct -j %s,%s,%s,%s,%s,%s --format=JobID,JobName,State,Elapsed,ExitCode\n' \
  "$EXP305_ARRAY" "$EXP305_FINAL" "$EXP306_ARRAY" "$EXP306_FINAL" "$EXP307_ARRAY" "$EXP307_FINAL"
