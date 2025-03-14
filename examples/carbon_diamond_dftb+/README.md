Instillation of DFTB should be done via the DFTB+ website.

- 1_C_Tersoff_fc_and_harmonic_properties.py proceeds as follows:
	
    1. Set up force constant object and compute 2nd, 3rd force constants using C.tersoff.

    2. Set up phonon object (5x5x5 k-point mesh) and perform quantum simulation at 300K.
     
    3. Compute and visualize harmonic properties (i.e. dispersion relation, group velocity and DOS). 
			      
-  2_C_Tersoff_thermal_conductivity.py proceeds as follows:

    1. Set up force constant object and compute 2nd, 3rd force constants using C.tersoff.

    2. Set up phonon object (5x5x5 k-point mesh) and perform quantum simulation at 300K.

    3. Set up Conductivity object and compute thermal conductivity with Relaxation Time Approximation (RTA), 
				self-consistent (sc) and direct inversion of scattering matrix (inverse) methods.

-  3_C_Tersoff_visualize_anharmonic_properties.py proceeds as follows:

    1. Compute and visualize phase spaces.
			
	2. Compute and compare phonon life times with RTA and direct inversion methods.


- To run this example, navigate to this directory and execute:

```python
python 1_C_Tersoff_fc_and_harmonic_properties.py
python 2_C_Tersoff_thermal_conductivity.py
python 3_C_Tersoff_visualize_anharmonic_properties.py
```
- To view figures generated during simulations, navigate to this folder: ***plots/5_5_5/***
- To access data computed during simulations, navigate to this folder: ***ALD_c_diamond***

Notes

If you are having problems with the dftb+ path not working, an easy way is to add it to path with the following.
```bash
export PATH=$PATH:/<path_to_DFTB+_dir">
```

