#!/bin/bash
printf "\n>>> kALDo Automated Comparison Tool (kactus v.0.2a) <<<\n"
printf "
#     ▄█   ▄█▄    ▄████████  ▄████████     ███     ███    █▄     ▄████████
#    ███ ▄███▀   ███    ███ ███    ███ ▀█████████▄ ███    ███   ███    ███
#    ███▐██▀     ███    ███ ███    █▀     ▀███▀▀██ ███    ███   ███    █▀
#   ▄█████▀      ███    ███ ███            ███   ▀ ███    ███   ███
#  ▀▀█████▄    ▀███████████ ███            ███     ███    ███ ▀███████████
#    ███▐██▄     ███    ███ ███    █▄      ███     ███    ███          ███
#    ███ ▀███▄   ███    ███ ███    ███     ███     ███    ███    ▄█    ███
#    ███   ▀█▀   ███    █▀  ████████▀     ▄████▀   ████████▀   ▄████████▀
#    ▀
#                       VERSION 0.1(a)
#             INTENDED FOR NANOTHEORY INTERNAL USE
#                 https://nanotheory.github.io
"

# Assumes modded harmonic_with_q.py file is installed in your kALDo build
# executables
matdyn="REPLACEME"
if [ ! -f ${matdyn} ];
then
    echo "Uh-oh! You need to replace the matdyn variable in this script"
    echo "with the absolute path to the moddified matdyn executable,"
    echo "Then try again! :)"
    echo "Make sure it's compiled with the modified matdyn.f90 file"
    echo "You can probably use this command:"
    echo "sed -i \"s/REPLACEME/<your-path>/\" kactus.sh"
    printf "\n\n"
    exit 1
fi
# directories
forcedir="forces"
# Data files
tarball="${forcedir}.tgz"


### py scripts ##########
# 0 - gen path
py_path='0_gen_path.py'
# 1 - run kaldo + matdyn
py_kpack='1_pack_kaldo.py'
py_mdpack='1_pack_md.py'
# 2 - analyze dynmat
py_frc="2_compare_frc.py"
printf "- Execution Plan -----\n\t${py_path}\n\tmatdyn.x & ${py_mdpack}"
printf "\n\t${py_kpack}\n\t${py_frc}\n\n"

# IO Files ##############
# 0_
ptxt="path.out.txt"
# 1_
# mdin="md.in.band_path" # Option 1
mdin="md.in.coord_path" # Option 2
mdout="md.out.txt"
mdpack="md.out.pack"
mdcomp="md.out.npy"
kcomp="kaldo.out.npy"
kpack="kaldo.out.pack"
# 2_
fout="frc.out.txt"
fpack="frc.out.npy"
# 3_
#wout="freq.out.txt"
#wpack="freq.out.npy"
printf "Input Files: ${mdin} + ${forcedir} (or ${tarball})\n"
printf "Selected Executable: ${matdyn}\n---\n"


printf "!! Step 0 - Path Configuration\n"
if [ ! -f ${ptxt} ];
then
  printf "\tGenerating path\n"
  printf "\tSection output files: ${ptxt}\n\tRunning python .."
  python ${py_path} &> /dev/null
  printf "\tdone!\n"
  printf "\tPath stored as text at ${ptxt}\n"
  printf "\tStep Complete! :) \n\n"
else
  printf "\tFound previous path config found at ${ptxt}\n"
  printf "\tStep Complete! :) \n\n"
fi

printf "!! Step 1 - Generate Espresso Data Along Path\n"
################################################################
##############  E  S  P  R  E  S  S  O  ########################
if [ ! -f ${mdcomp} ];
then
  if [ ! -d ${forcedir} ]; # untar
  then
    printf "\tUntar forces to ${forcedir} .."
    tar -xvzf ${tarball}
    printf "\tdone!\n"
  else
    printf "\tForces found at ${forcedir}\n"
  fi

  if [ ! -f ${mdin} ];
  then
    printf "\tCreate matdyn input ${mdin} from ${mdin}.tmp .."
    cat ${mdin}.tmp > ${mdin}
    printf "\n$( cat ${ptxt} | wc -l )\n" >> ${mdin}
    cat ${ptxt} >> ${mdin}
    printf "\tdone!\n"
  else
    printf "\tFound previous matdyn input at ${mdin}\n"
  fi

  printf "\tRunning ${matdyn} .."
  ${matdyn} < ${mdin} &> ${mdout}
  exitcomp=$?
  printf "\t done!\n"
  if [ ${exitcomp} != 0 ];
  then
    printf "\n\n!! Matdyn failed\n\n${mdout} --\n"
    cat ${mdout}
    printf "\n--\n\n"
    printf ">>> Exiting kactus <<<\n\n"
    exit ${exitcomp}
  fi

  printf "\tRunning ${py_mdpack} .."
  python ${py_mdpack} &> ${mdpack}
  exitcomp=$?
  printf "\tdone!\n"
  if [ ${exitcomp} != 0 ];
  then
    printf "\n\n!! Python failed on ${py_mdpack}\n\nPrinting relevant log file:\n${mdpack} --\n"
    cat ${mdpack}
    printf "\n--\n\n"
    printf ">>> Exiting kactus <<<\n\n"
    exit ${exitcomp}
  fi
  printf "\tStep Complete! :) \n\n"
else
  printf "\tFound previously generated ${mdcomp}!\n"
  printf "\tStep Complete! :) \n\n"
fi
################################################################
################################################################

printf "!! Step 2 - Run kALDo Along Same Path\n"
################################################################
##############  k  A  L  D  O  #################################
if [ ! -f ${kcomp} ];
then
  if [ ! -d ${forcedir} ]; # untar
  then
    printf "\tUntar forces to ${forcedir} .."
    tar -xvzf ${tarball}
    printf "\tdone!\n"
  else
    printf "\tForces found at ${forcedir}\n"
  fi

  printf "\tRunning kaldo .."
  python ${py_kpack} &> ${kpack}
  exitcomp=$?
  printf "\tdone!\n"
  if [ ${exitcomp} != 0 ];
  then
    printf "\n\n!! Python failed to run ${py_kpack}\n\nPrinting relevant log file:\n${kpack} --\n"
    cat ${kpack}
    printf "\n--\n\n"
    printf ">>> Exiting kactus <<<\n\n"
    exit ${exitcomp}
  fi
  printf "\tStep Complete! :) \n\n"
else
  printf "\tFound previously generated ${kcomp}! Continuing.\n"
  printf "\tStep Complete! :) \n\n"
fi
################################################################
################################################################

printf "!! Step 3 - Compare Forces Along Path\n"
################################################################
##### C O M P A R E ############################################
# Attempt to run comparison
printf "\tRunning comparison script .."
python ${py_frc} &> ${fout}
exitcomp=$?
printf "\tdone!\n"
if [ ${exitcomp} == 0 ];
then
  if [ -f ${fpack} ];
  then''
    printf "!! Final Verdict: Guilty.\n"
    printf "!! Mismatches detected in at least one category\n"
    printf "!! Mismatches were not fatal\n"
    printf "!! Check text output: vi ${fout}\n"
    printf "!! Check object arrays stored at ${fpack}\n"
  else
    printf "!! Final Verdict: Innocent.\n"
    printf "!! No mismatches detected in any category\n"
  fi
  printf "\n>>> kactus ran succesfully <<<\n\n"
else
  printf "\n\n!! Python failed to run ${py_frc}\n\nPrinting relevant log file:\n${fout} --\n"
  cat ${fout}
  printf "\n--\n\n"
  printf "\n\n>>> Exiting kactus <<<\n\n"
  exit ${exitcomp}
fi
################################################################
################################################################
exit 0
