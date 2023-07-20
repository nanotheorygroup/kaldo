#!/bin/bash
# Instructions:
# Run the script with your preferred flags to setup forces if it hasn't
# been done, and launch kaldo with the harmonic flag detected by environment variable.
# "0_batch_run.sh -p"
# -p/--parallel if you can launch parallel processes safely


printf "\n\n\n\tRunning monolayer example :)\n"
#################################################
# Setup
printf "\nEnsuring force constants are ready.."
for calculator in "hiphive" "espresso"
do
  if [ ! -d "${calculator}_fcs" ];
  then
    # untar HiPhive force constants
    printf "\n\tInflating tarball.."
    tar -xvzf tools/${calculator}_fcs.tgz
    printf "\tUnpacked ${calculator}!!"
  fi
done
printf "\tYep!\n\n"

printf "\nEnsuring directories are ready.."
for dirs_to_setup in ${kaldo_outputs} ${kaldo_ald}
do
  if [ ! -d ${dirs_to_setup} ];
  then
    mkdir ${dirs_to_setup}
  fi
done
printf "\tDone!! Setup complete proceeding to kALDo run.\n\n\n"

case "$1" in
  -p|--parallel)
    printf "Launching kALDo as a disowned process. Process is disowned from your shell and "
    printf "output is redirected to ${kaldo_outputs}/output.run_kaldo"
    python 1_run_kaldo.py ${kaldo_harmonicflag} &> ${kaldo_outputs}/output.run_kaldo & disown
    ;;
  *)
    printf "Launching kALDo as a child of this shell. Output is redirected, but you must leave "
    printf "this shell running until the simulation is complete."
    python 1_run_kaldo.py ${sys} ${kaldo_harmonicflag} &> ${kaldo_outputs}/output.run_kaldo
    ;;
esac
printf "\n\n"