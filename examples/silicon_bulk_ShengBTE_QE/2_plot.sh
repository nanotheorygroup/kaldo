# Quantities to plot
plot_quants=('bandwidth' 'phase_space' 'heat_capacity' 'population')

printf "Plotting! --\n"
# note vd arg passed to 2_plot_dispersion.py is to indicate you want velocity+dispersion plotted
case "$1" in
  -p|--parallel)
    python 3_plot_dispersions.py vd &> ${kaldo_outputs}/plot_dispersion.out & disown
    ;;
  *)
    python 3_plot_dispersions.py vd &> ${kaldo_outputs}/plot_dispersion.out
    ;;
esac
if [ "${kaldo_harmonicflag}" == "harmonic" ];
then
  printf "\nComplete!\n"
  exit 0
fi
for quantity in ${plot_quants[@]}
do
  case "$1" in
    -p|--parallel)
      python 4_plot_phonon_data.py ${quantity} &> ${kaldo_outputs}/plot_${quantity}.out & disown
      ;;
    *)
      python 4_plot_phonon_data.py ${quantity} &> ${kaldo_outputs}/plot_${quantity}.out
      ;;
  esac
done
printf "\nComplete!\n"
exit 0
