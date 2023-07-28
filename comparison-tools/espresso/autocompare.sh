#!/bin/bash
# Assumes modded harmonic_with_q.py file is installed in your kALDo build
# directories
# home=$( realpath . )
md_mods="modded_espresso"
k_mods="modded_kaldo"
tarball="forces.tgz"

# scripts
py_kpack='pack-kaldo.py'
py_mdpack='pack-md.py'
py_path='path-gen.py'
py_compare="path-compare.py"

# Data files
# mdin="md.in.band_path" # Option 1
mdin="md.in.coord_path" # Option 2
mdout="md.out.txt"
mdpack="md.out.pack"
mdcomp="md.out.npy"

kcomp="kaldo.out.npy"
kpack="kaldo.out.pack"

ptxt="path.out.txt"

compare="mismatch.txt"
finalout="mismatch.npy"

# executables
matdyn="/home/nwlundgren/develop/quantumespresso/build-6.21.2023/bin/matdyn.x"

printf "\n\n!!\tWorking on comparisons \n"
#########################################################################
##############  E  S  P  R  E  S  S  O  ########################
if [ ! -f ${mdcomp} ];
then
  printf "! Attempting to package QE data ..."
  if [ ! -f espresso.ifc2 ]; # untar
  then
    tar -xvzf ${tarball}
  fi

  python ${py_path}
  cat ${mdin}.tmp > ${mdin}
  printf "$( cat ${ptxt} | wc -l )\n" >> ${mdin}
  cat ${ptxt} >> ${mdin}

  ${matdyn} < ${mdin} &> ${mdout}
  if [ $( wc -l < ${mdout}) -lt 5000 ];
  then # if the output is small something went wrong
    printf "\n\tUnusual matdyn.x behaviour, fatal error\n\n"
    exit 1
  fi

  python ${py_mdpack} > ${mdpack}
  printf " complete! :) \n"
fi
################################################################



##############  k  A  L  D  O  #################################
if [ ! -f ${kcomp} ];
then
  printf "! Attempting to package comparable kALDo data ..."
  if [ ! -f ${mdcomp} ];
  then
    printf "\n\tMissing matdyn output, no q-pt list to compare\n\n"
    exit 1
  fi

  python ${py_kpack} &> ${kpack}
  printf " complete! :) \n"
fi
################################################################
########################################################################

printf "\n! We think data for both programs has been output and packaged uniformly\n"
printf "Attempting to run comparison script .. "

# Attempt to run comparison
python ${py_compare} > ${compare}

printf "\tComparison has run and exited!\n\tOutput can be found at out.py.comparison\n"
printf "\tMismatches saved to ${finalout}\n"
printf "\n\n"
exit 0






