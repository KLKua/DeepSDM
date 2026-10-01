#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-hmsc_conf.yml}"
MAX_JOBS="${MAX_JOBS:-4}"
BATCH_SIZE=5
N_SPECIES=121

for START in $(seq 1 "${BATCH_SIZE}" "${N_SPECIES}"); do
  while [ "$(jobs -rp | wc -l)" -ge "${MAX_JOBS}" ]; do
    wait -n
  done
  Rscript evaluate_three_models_constantthreshold_hmsc.R "${START}" "${CONFIG}" &
done
wait
