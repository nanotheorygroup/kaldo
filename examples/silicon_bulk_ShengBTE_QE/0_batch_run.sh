#!/bin/bash
# Instructions:
# Edit the variables in the run parameters section to
# reflect your preferences
# Run the script with your preffered flags
# "0_batch_run.sh -p"
# -p/--parallel if you can launch parallel processes safely


## Run Parameters ################################
## Unpack silicon-qe data + create directories
setup="true"

## Calculate or plot?
run_type="calculate"

# Do we want Kappa? (~ 1 hour per system)
harmonic_only="true"

# Quantities to plot
plot_quants=('bandwidth' 'phasespace' 'participationratio')

#################################################

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
    mkdir ${kaldo_outputs}
  fi

  printf "## Running kALDo! --\n"
  if [ "${harmonic_only}" == "true" ];
  then
    harmonicflag="harmonic"
  fi

  for sys in ${systems_to_run[@]}
  do
    printf "\tLaunching calculations on ${sys}\n"
    printf "\t\tpython 1_run_kaldo.py ${sys} ${dispersionflag}\n"
    case "$1" in
      -p|--parallel)
        python 1_run_kaldo.py ${sys} ${harmonicflag} &> ${kaldo_outputs}/${sys}.out & disown
        printf "\t\tLaunch succesful\n"
        ;;
      *)
        python 1_run_kaldo.py ${sys} ${harmonicflag} &> ${kaldo_outputs}/${sys}.out
        ;;
    esac
  done
elif ${run_type}=='plot'
then
  printf "Plotting! --\n"
  # note vd arg passed to 2_plot_dispersion.py is to indicate you want velocity+dispersion plotted
  case "$1" in
    -p|--parallel)
      python 2_plot_dispersion.py vd ${systems_to_run[@]} &> ${kaldo_outputs}/plot_dispersion.out & disown
      ;;
    *)
      python 2_plot_dispersion.py vd ${systems_to_run[@]} &> ${kaldo_outputs}/plot_dispersion.out
      ;;
  esac
  if [ "${harmonic_only}" == "true" ];
  then
    printf "\nComplete!\n"
    quit(0)
  else
  then
    for quantity in plot_quants
    do
      case "$1" in
        -p|--parallel)
          python 3_plot_phonon_data.py ${quantity} ${systems_to_run[@]} &> ${kaldo_outputs}/plot_dispersion.out & disown
          ;;
        *)
          python 3_plot_phonon_data.py ${quantity} ${systems_to_run[@]} &> ${kaldo_outputs}/plot_dispersion.out
          ;;
      esac
    done
  fi
fi
printf "\nComplete!\n"
quit(0)
