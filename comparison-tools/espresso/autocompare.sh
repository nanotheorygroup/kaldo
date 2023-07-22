#!/bin/bash
# Assumes modded harmonic_with_q.py file is installed in your kALDo build
# directories
home=$( realpath . )
md_mods="modded_espresso"
k_mods="modded_kaldo"
workup="workup"

# scripts
py_kpack='repack-kaldo.py'
py_mdpack='repack-matdyn.py'
py_compare="output_comparison_arrays.py"

# Data files
mdout="output.md"
mdcomp="compiled-matdyn.npy"
kcomp="compiled-kaldo.npy"

# executables
matdyn="/home/nwlundgren/develop/quantumespresso/build-debug/bin/matdyn.x"

printf "\n\n\t\t Working on comparisons \n"
#########################################################################
##############  E  S  P  R  E  S  S  O  ########################
if [ ! -f ${workup}/${mdcomp} ];
then
  printf "! Attempting to package QE data ..."
  if [ ! -f ${workup}/espresso.ifc2 ]; # untar
  then
    tar -xvzf ${workup}
  fi
  cp ${md_mods}/${py_mdpack} ${workup}/
  cd ${workup}

  ${matdyn} < input.md &> ${mdout}
  if [ $( wc -l < ${mdout}) -lt 5000 ];
  then # if the output is small something went wrong
    printf "\n\tUnusual matdyn.x behaviour, fatal error\n\n"
    exit 1
  fi

  python ${py_mdpack} > output.rpk
  printf " complete! :) \n"
  cd ${home}
fi
################################################################



##############  k  A  L  D  O  #################################
if [ ! -f ${workup}/${kcomp} ];
then
  printf "! Attempting to package comparable kALDo data ..."
  if [ ! -f ${workup}/${mdcomp} ];
  then
    printf "\n\tMissing matdyn output, no q-pt list to compare\n\n"
    exit 1
  fi
  cp ${k_mods}/${py_krun_pack}  ${workup}/
  cp ${k_mods}/${py_kpack} ${workup}/
  cd ${workup}

  python ${py_krun} > output.krn
  python ${py_kpack} > output.kpk
  printf " complete! :) \n"
  cd ${home}
fi
################################################################
########################################################################

printf "\nWe think data for both programs has been output and packaged uniformly\n"
cd ${workup}
# Attempt to run comparison
python ${py_compare} > ${home}/output.comparison
cd ${home}
printf "Comparison has run and exited. Output can be found at ${home}/output.comparison\n"
exit 0







