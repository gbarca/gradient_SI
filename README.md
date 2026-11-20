# Supporting Information

This repository contains input xyzs, output files, and plotting scripts used to generate figures and tables for the paper's HF gradient benchmarks.

## Contents

- [input](input): Molecular input structures.
  - [input/water](input/water)
  - [input/gly](input/gly)
  - [input/misc](input/misc)
- [output](output): Raw program outputs organized as:
  - output/{code}/{basis}/{molecule}{size}_run{1,2,3}.out (three timing runs per system)
  - output/{code}/misc/{system}.out (large molecules, def2-SVP, single run)
  - where code is one of: exess (EXESS), exess_ri (EXESS RI), gpu4pyscf (GPU4PySCF), quick, terachem, veloxchem, cuest_direct (cuEST direct J/K), cuest_gridk (cuEST Grid-K: non-incremental Fock build, SAD-reference overlap-fitted exchange on the SG-0(-2) grid)
  - output/accurate_cart and output/accurate_sph: tight-threshold GPU4PySCF reference gradients (Cartesian / spherical)
  - Basis directories replace `*` with `d` (6-31G* -> 6-31Gd, 6-31G** -> 6-31Gdd)
- [performance_output](performance_output): Performance plots, CSV, and LaTeX tables.
- [accuracy_output](accuracy_output): Accuracy CSV and LaTeX tables.
- [read_output.py](read_output.py): Shared parsing and LaTeX table helper utilities.

## Reproducing Performance Outputs

From repository root:

```bash
python3 compare_performance.py
```

This writes outputs to [performance_output](performance_output), including:

- performance_output/gradient_timings.csv and misc_gradient_timings.csv
- performance_output/gradient_timings_tables.tex and misc_gradient_timings_table.tex
- PNGs named gradient_time_{molecule}_{basis}.png

## Reproducing Accuracy Outputs

From repository root:

```bash
python3 compare_accuracy.py
```

This writes outputs to [accuracy_output](accuracy_output):

- accuracy_output/gradient_rmsd.csv
- accuracy_output/gradient_rmsd_tables.tex

## Notes

- 6-31G* and 6-31G** are run with Cartesian basis functions; cc-pVDZ, def2-SVP, cc-pVTZ and def2-TZVP with spherical ones. QUICK is Cartesian-only and VeloxChem is spherical-only, so they use their own convention for every basis.
- Each system has three runs (timings take the minimum, accuracy results come from run 1).
- EXESS (RI) and cuEST (Grid-K) are left out of the accuracy tables, since the RI and Grid-K approximations introduce their own error; the mean Grid-K RMSD is printed by compare_accuracy.py.
- Systems that ran out of GPU memory or whose SCF did not converge keep a single run as evidence; they are marked OOM / NC in the tables and CSVs and skipped in the plots.
