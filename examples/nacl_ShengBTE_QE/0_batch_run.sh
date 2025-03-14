#!/bin/bash
# tip: Please source tools/vars.sh before running this script!
# We set a few environment variables helpful to keep it organized.

# Instructions:
# Edit the variables in the run parameters section to
# reflect your preferences
# Run the script with your preffered flags
# use chmod +x to make this script executable and then: "./0_batch_run.sh"
# or "bash ./0_batch_run.sh"

# Dispersions:
# If you want to exit after generating the phonon dispersion, please
# set the environment variable "kaldo_harmonicflag" to "harmonic" like
# export kaldo_harmonicflag="harmonic" or run python script 1 with "harmonic"
# as an argument

#################################################
# Setup
if [ ! -f "${kaldo_inputs}/with_charges/POSCAR" ];
then
  # untar Silicon force constants
  printf "Inflating forces..\t"
  mkdir ${kaldo_inputs}
  tar -xvzf nacl-sheng-qe.tgz -C ${kaldo_inputs}

  # now we make a second copy without charge data to compare
  # the sed command deletes the rows with charge information (7-17)
  # as well as switching the T(rue) to F(alse)
  printf "done!\nCreating copy without charges.."
  cp -r ${kaldo_inputs}/with_charges ${kaldo_inputs}/no_charges
  sed -i "7,17d; s/T/F/" ${kaldo_inputs}/no_charges/espresso.ifc2
  printf "\tall set!\n\n"
fi

if [ ! -d ${kaldo_outputs} ];
then
  # create directory for output files
  printf "Creating directories..\n"
  mkdir ${kaldo_outputs} # for plots and logs
  mkdir ${kaldo_ald} # for intermediate data
fi
##################################################

## Running! ##
printf "## Running kALDo! --\n"
printf "\tLaunching thermal conductivity calculation:"
printf "\t\tpython 1_run_kaldo.py ${kaldo_harmonicflag}\n"
python 1_run_kaldo.py nac ${kaldo_harmonicflag} &> ${kaldo_outputs}/out.kaldo_run & disown
python 1_run_kaldo.py ${kaldo_harmonicflag} &> ${kaldo_outputs}/out.kaldo_run & disown
printf "\tProcess Launched. Exiting bash script\n"
