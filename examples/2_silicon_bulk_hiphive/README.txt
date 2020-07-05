1. This example requires the installation of hiphive package. Please run the following command to install hiphive

   pip install hiphive	

2. After compiling lammps as python library (please see the documentation for this set up), please run the following 
   commands in orders to run the example:

   wget http://sophe.ucdavis.edu/structures/forcefields.zip
   unzip forcefields.zip	
    
   python 1_generate_fcs.py
   python 2_calculate_phonons_si.py
