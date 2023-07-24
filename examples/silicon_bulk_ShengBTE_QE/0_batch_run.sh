#!/bin/bash
# Instructions:
# Edit the variables in the run parameters section to
# reflect your preferences
# Run the script with your preffered flags
# "0_batch_run.sh -p"
# -p/--parallel if you can launch parallel processes safely

# Parameters
# match this to kaldo_systems environ var
systems_to_run=('3u' '3n' '8n' '8u')

#################################################
# Setup
if [ ! -f "${kaldo_inputs}/3x3x3/POSCAR" ];
then
  # untar Silicon force constants
  printf "Inflating forces..\n"
  tar -xvzf tools/si-sheng-qe.tgz
  mkdir ${kaldo_inputs}
  for supercells in 1x1x1 3x3x3 5x5x5 8x8x8
  do
    mv ${supercells} ${kaldo_inputs}/
  done
  printf "\tUnpacked!\n\n"
fi
if [ ! -d ${kaldo_outputs} ];
then
  # create directory for output files
  printf "Creating directories..\n"
  mkdir ${kaldo_outputs}
  mkdir ${kaldo_ald}
fi

# Run
printf "## Running kALDo! --\n"
for sys in ${systems_to_run[@]}
do
  printf "\tLaunching calculations on ${sys}\n"
  printf "\t\tpython 1_run_kaldo.py ${sys} ${kaldo_harmonicflag}\n"
  case "$1" in
    -p|--parallel)
      python 1_run_kaldo.py ${sys} ${kaldo_harmonicflag} &> ${kaldo_outputs}/${sys}.out & disown
      printf "\t\tLaunch succesful\n"
      ;;
    *)
      python 1_run_kaldo.py ${sys} ${kaldo_harmonicflag} &> ${kaldo_outputs}/${sys}.out
      ;;
  esac
done
