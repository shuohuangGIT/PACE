# PACE (Planet population synthesis and their Architecture evolution in star Cluster Environments)
## 1. Code description:
Population synthsis code based on Venice

## 2. File description
Run python3 venice_pps_run.py to run the default pps model. Parameters are basically set in this file. We trace the planets and save them in file planet_data.npz, run venice_pps_plot.py to plot the data.
The files named module*.py are building different physical processes. For example, module_migration.py builds migration process. (is the name module good?)
The file extra_funcs.py defines some extra functions used in the files. parames.py defile some constants used in OL18.py.

However, venice.py, symmetric_matrix.py are src files of Venice (Wilhlem et al. in prep). OL18.py is the source file to calculate pebble accretion \citep{Ormel & Liu 2018}.

## 3. Test run
key file for vader: PACE_vader/PACE/amuse_vader_src/vader should be in the amuse vader community (amuse/src/amuse/community/vader/src/prob). The worker file should be included. Maybe the easiest (but most risky) way is to replace the vader file in the community.
compile: make vader.code
If succeed, run: python3 test_run.py. In principle, it should form planet. You can monitor the growth and migration of planet by compile read_planet_evo.py
Key function in the python script: run_single_pps (locates in venice_pps_setup_pebb_vader_OL18.py)

