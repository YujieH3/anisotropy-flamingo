# Anisotropy FLAMINGO Project
![Work in Progress](https://img.shields.io/badge/status-in%20progress-yellow)

The project study the anisotropy of the universe with galaxy scaling relations. We use the FLAMINGO simulation to create thousands of mock lightcones and see what level of cosmic anisotropy we can expect in a LCDM universe. 

The work is a continuation of observational studies from:
- [Migkas et al. 2020](https://ui.adsabs.harvard.edu/abs/2020A%26A...636A..15M/abstract)
- [Migkas et al. 2021](https://ui.adsabs.harvard.edu/abs/2021A%26A...649A.151M/abstract)

The universe might look more like a 🌰 than a 🎱.

## To-Do List
- [ ] Write a batch script to run bootstrapping without fixing directions but scan a small region.
- [ ] Analyse the results of mcmc h0
- [ ] Run MCMC bulk flow analysis
- [ ] Write a batch script to run chi-square bulk flow fixing directions
- [ ] Submit chi-square bulk flow fixing directions on the cluster
- [ ] Write a batch script to run chi-square bulk flow without fixing directions
- [ ] Submit chi-square bulk flow without fixing directions on the cluster
- [ ] Wrtie a small script to examine the completeness of halo_lightcones: 19 columns of quantities in four snapshots
- [ ] Make a cron job to make samples every hour.
- [x] Submit batch_best_fit.sh
- [x] Submit H0 chi-square + bootstrapping on the cluster.
- [x] Write a batch script to run the H0 chi-square + bootstrapping analysis on the cluster.
- [x] Rewrite scan-bootstrap to scan-bootstrap-fix-lonlat to bootstrap only on max difference of lon and lat.
- [x] Submit H0 MCMC analysis on the cluster.
- [x] Write a batch script to run the H0 MCMC analysis on the cluster.
- [x] lightcone0000 - 0004 are corrupted, remember to redo the sample creation on them.
- [x] Make scripts/_1_combine_lightcone.py, scripts/_2_band_patch.py, scripts/_3_rotate_lightcone.py take individual files as input instead of a directory, easier to run in parallel with lightcone creation script (scripts/__fast_make_lightcone_mpi.py).
- [x] Set up a cron job to copy to backup for further operations every hour.
- [x] Lightcone debugged and running on the cluster.

## Warning

This repository is a **working directory** for my ongoing research on the anisotropy of the universe using the FLAMINGO simulation. It is **not intended as a fully-developed package** or tool for general use. The code here is primarily for personal experimentation and may not be well-documented, optimized, or suitable for reuse by others.

If you’re looking for a polished and reusable solution, this repository may not meet your expectations. However, you are welcome to explore the content and reach out with any questions, suggestions, or collaborations!