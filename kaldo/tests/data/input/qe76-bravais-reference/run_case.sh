#!/usr/bin/env bash
# Reproduce one plain-text QE 7.6 q2r/matdyn reference case.
set -euo pipefail

qe_bin=${QE_BIN:?set QE_BIN to the Quantum ESPRESSO 7.6 bin directory}
mpi_command=${MPI_COMMAND:-mpirun -n 2}
case_dir=${1:?usage: run_case.sh CASE_DIRECTORY}
case_dir=$(CDPATH= cd -- "$case_dir" && pwd)
case_id=${case_dir##*/}
read -r -a mpi_parts <<< "$mpi_command"

cd "$case_dir"
test -d pseudo || {
  echo "populate $case_dir/pseudo using pseudopotentials.json" >&2
  exit 2
}
mkdir -p out

"${mpi_parts[@]}" "$qe_bin/pw.x" -in scf.in > scf.out
"${mpi_parts[@]}" "$qe_bin/ph.x" -in ph-gamma.in > ph-gamma.out
"${mpi_parts[@]}" "$qe_bin/ph.x" -in ph-ldisp.in > ph-ldisp.out
"${mpi_parts[@]}" "$qe_bin/q2r.x" -in q2r.in > q2r.out

python -c \
  "from kaldo.interfaces.qe_io import strip_q2r_nac; strip_q2r_nac('$case_id.fc', '$case_id.without-nac.fc')"
"${mpi_parts[@]}" "$qe_bin/matdyn.x" -in matdyn-on.in > matdyn-on.out
"${mpi_parts[@]}" "$qe_bin/matdyn.x" -in matdyn-off.in > matdyn-off.out
