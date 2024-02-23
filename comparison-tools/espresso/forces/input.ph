InSe 8 8 1 phonons
&INPUTPH
   outdir = 'tmp/'
   prefix = 'mono'
   fildyn = 'DYN.mat'
   tr2_ph = 1.0000000000d-18
   alpha_mix(1) = 0.7
   verbosity = 'high'
   nmix_ph = 5
   epsil = .true.   
   ldisp = .true.
   nq1 = 9, nq2 = 9, nq3 = 1
   recover=.true.
/
