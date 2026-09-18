#!/usr/bin/env bash
# Compile and run the fixture against an *unmodified* QE rigid.f90.
set -euo pipefail
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
qe_rigid=${QE_RIGID_F90:?set QE_RIGID_F90 to qe-7.6/PHonon/PH/rigid.f90}
out_dir=${1:?usage: run.sh OUTPUT_DIRECTORY}
expected=c2da58dfa6849c4edb6f96ad8ad2be58834ae23142bdac2ee33eeebf1eb21e88
test -f "$qe_rigid"
test "$(sha256sum "$qe_rigid" | awk '{print $1}')" = "$expected"
mkdir -p "$out_dir"
gfortran -O0 -fno-fast-math -ffp-contract=off -ffunction-sections -fdata-sections \
  -ffree-line-length-none -J"$out_dir" -I"$out_dir" \
  "$here/kinds_constants.f90" "$qe_rigid" "$here/qe76_rigid_f90_driver.f90" \
  -Wl,--gc-sections -o "$out_dir/qe76_rigid_f90_driver"
"$out_dir/qe76_rigid_f90_driver" "$here/rigid_fixture.nml" "$out_dir/rgd_blocks.dat"
python3 "$here/generate_reference.py" "$out_dir/rgd_blocks.dat" "$out_dir/qe76_rigid_f90_reference.npz"
