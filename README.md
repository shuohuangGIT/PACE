# venice-population-syntheis
## 1. Code description:
Population synthsis code based on Venice

## 2. File description
Run python3 venice_pps_run.py to run the default pps model. Parameters are basically set in this file. We trace the planets and save them in file planet_data.npz, run venice_pps_plot.py to plot the data.
The files named module*.py are building different physical processes. For example, module_migration.py builds migration process.
The file extra_funcs.py defines some extra functions used in the files. 

However, venice.py, symmetric_matrix.py are src files of Venice.
