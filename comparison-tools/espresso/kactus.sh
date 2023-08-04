#!/bin/bash
printf "\n>>> kALDo Automated Comparison Tool (kactus v.0a) <<<\n"
printf "\t          __ ___       __\n\t|_/  /\\  /    |  /  \\ (_\n\t| \\ /--\\ \\__  |  \\__/ __)\n"
printf "\t        __              __\n\t       /  \    /\  |   |__) |__|  /\ \n"
printf "\t\/ .   \__/   /--\ |__ |    |  | /--\ \n"
printf "\n"

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
printf "Execution Plan --\n\t${py_path}\n\tmatdyn.x & ${py_mdpack}"
printf "\n\t${py_kpack}\n\t${py_frc}\n--\n\n"

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
#wout="freq.out.txt"
#wpack="freq.out.npy"
printf "Input Files: ${mdin} + ${forcedir} (or ${tarball})\n"
printf "Selected Executable: ${matdyn}\n"


printf "!! Step 0 - Path Configuration\n"
if [ ! -f ${ptxt} ];
then
  printf "\tGenerating path\n"
  printf "\tSection output files: ${ptxt}\n\tRunning python .."
  python ${py_path} &> /dev/null
  printf "\tdone!\n"
else
  printf "\tPath config found at ${ptxt}\n"
fi

printf "!! Step 1 - Generate Espresso Data Along Path\n"
#########################################################################
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
  printf "\t done!\n"
  if [ $( wc -l < ${mdout}) -lt 5000 ];
  then # if the output is small something went wrong; exit
    printf "!! Unusual matdyn.x behaviour, fatal error\n"
    printf ">>> kactus failed, exiting .. <<<\n\n"
    exit 1
  fi

  python ${py_mdpack} &> ${mdpack}
  printf "\tStep Complete! :) \n"
else
  printf "\tFound previously generated ${mdcomp}! Continuing.\n"
fi
################################################################


printf "!! Step 2 - Run kALDo Along Same Path\n"
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
  printf "\tdone!\n"
  if [ ! -f ${kcomp} ];
  then
    printf "!! Failed to generate ${kcomp}, check ${kpack} log file\n"
    printf "\n>>> kactus failed, exiting .. <<<\n\n"
    exit 1
  fi
  printf "Step Complete! :) \n"
else
  printf "\tFound previously generated ${kcomp}! Continuing.\n"
fi
################################################################
########################################################################

printf "!! Step 3 - Compare Forces Along Path\n"
# Attempt to run comparison
printf "\tRunning comparison script .."
python ${py_frc} &> ${fout}
exitcomp=$?
printf "\tdone!\n"


if [ ${exitcomp} == 0 ];
then
  if [ -f ${fpack} ];
  then
    printf "!! Final Verdict: Guilty.\n"
    printf "!! Mismatches detected in at least one category\n"
    printf "!! Mismatches were not fatal\n"
  else
    printf "!! Final Verdict: Innocent.\n"
    printf "!! No mismatches detected in any category\n"
  fi
  printf "\n>>> kactus ran succesfully <<<\n\n"
else
  printf "!!"
  printf "\n>>> kactus failed, check logs <<<\n\n"
fi
exit 0