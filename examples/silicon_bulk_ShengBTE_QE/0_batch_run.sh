#!/bin/bash
# Set run_type to either calculate or plot
run_type="calculate"
dispersion_only="false"
parallel="true"
systems_to_run=('3u', '3n', '8u', '8n')

if ${run_type}=="calculate"
then
  echo Running kALDo! --
  # untar Silicon force constants
  tar -xvzf si-qe-sheng-dat.tgz

  # create directory for output files
  export outputdir=kaldo-outputs
  mkdir ${outputdir}

  disownflag==""
  if ${parallel}=="true"
  then
    disownflag=="& disown"
  fi
  dispersionflag=""
  if ${dispersion_only}=="true"
  then
    dispersionflag="disp"
  fi
  for sys in ${systems_to_run[@]}
  do
          echo Launching calculations on ${sys}
          python 1_run_kaldo.py ${sys} ${dispersionflag} &> ${outputdir}/${sys}.out ${disownflag}
  done
elif ${run_type}=='plot'
then
  echo Plotting! --
  python 2_plot_dispersion.py ${systems_to_run[@]}
  python 2_plot_dispersion.py ${systems_to_run[@]} bandwidth
  python 2_plot_dispersion.py ${systems_to_run[@]} phasespace
  python 2_plot_dispersion.py ${systems_to_run[@]} participation_ratio
fi