#!/usr/bin/env bash
set -euo pipefail

case_name=${1:?usage: run_reference.sh CASE OUTPUT_DIRECTORY}
output_directory=${2:?usage: run_reference.sh CASE OUTPUT_DIRECTORY}
shengbte_binary=${SHENGBTE_BIN:-${HOME}/programs/shengbte/ShengBTE}
mpi_ranks=${SHENGBTE_MPI_RANKS:-2}

if [[ -e $output_directory ]]; then
    echo "output directory already exists: $output_directory" >&2
    exit 64
fi
if [[ ! -x $shengbte_binary ]]; then
    echo "ShengBTE executable is missing: $shengbte_binary" >&2
    exit 69
fi

reference_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
test_root=$(cd -- "$reference_root/../../.." && pwd)
mkdir -p -- "$output_directory"
cp -- "$reference_root/$case_name/CONTROL" "$output_directory/CONTROL"

case $case_name in
    vasp)
        cp -- "$test_root/si-crystal/vasp/FORCE_CONSTANTS" \
            "$output_directory/FORCE_CONSTANTS_2ND"
        cp -- "$test_root/si-crystal/vasp/FORCE_CONSTANTS_3RD" \
            "$output_directory/FORCE_CONSTANTS_3RD"
        ;;
    qe)
        cp -- "$test_root/si-crystal/qe/espresso.ifc2" \
            "$output_directory/espresso.ifc2"
        cp -- "$test_root/si-crystal/qe/FORCE_CONSTANTS_3RD" \
            "$output_directory/FORCE_CONSTANTS_3RD"
        ;;
    *)
        echo "unknown case $case_name; choose vasp or qe" >&2
        exit 64
        ;;
esac

cd -- "$output_directory"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
mpirun -n "$mpi_ranks" "$shengbte_binary"
