#!/bin/bash
# unpack silicon-qe data?
setup="true"

# can we launch multiple processes?
parallel="true"

# Calculate or plot?
run_type="calculate"

# Do we want Kappa? (~ 1 hour per system)
harmonic_only="true"

# Systems to compare
systems_to_run=('3u' '3n' '8n')

# where to put output
outputdir="kaldo-outputs"

if [ "${run_type}" == "calculate" ];
then
  printf "## Preparing batch kALDo launch ###\n"
  if [ "${setup}" == "true" ];
  then
    # untar Silicon force constants
    printf "Unpacking forces.."
    tar -xvzf si-qe-sheng-dat.tgz
    printf "\tUnpacked!\n\n"

    # create directory for output files
    printf "Creating directories.."
    mkdir ${outputdir}
  fi

  printf "## Running kALDo! --\n"
  if [ "${harmonic_only}" == "true" ];
  then
    harmonicflag="harmonic"
  fi

  for sys in ${systems_to_run[@]}
  do
          printf "\tLaunching calculations on ${sys}\n"
          printf "\t\tpython 1_run_kaldo.py ${sys} ${harmonicflag}\n"
          python 1_run_kaldo.py ${sys} ${harmonicflag} &> ${outputdir}/${sys}.out & disown
  done
  printf "\n\n### Launch succesful ###\n"
elif ${run_type}=='plot'
then
  echo Plotting! --
  python 2_plot_dispersion.py ${systems_to_run[@]}
  python 2_plot_dispersion.py ${systems_to_run[@]} bandwidth
  python 2_plot_dispersion.py ${systems_to_run[@]} phasespace
  python 2_plot_dispersion.py ${systems_to_run[@]} participation_ratio
fi
