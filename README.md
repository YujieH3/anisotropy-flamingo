# Cosmic anisotropy analysis with galaxy clusters

[![arXiv](https://img.shields.io/badge/arXiv-2504.01745-b31b1b.svg)](https://arxiv.org/abs/2504.01745)

This repo is a collection of the scripts used to analyse cosmic anisotropy using
galaxy clusters in [He et al. 2025](https://arxiv.org/abs/2504.01745). It was
applied to simulation data but can be extended to real galaxy cluster
catalogues. In fact we are extending the code to real data. You can find the
branch named 2504.01745, the archived version of this repo that made the 
calculation used in [He et al. 2025](https://arxiv.org/abs/2504.01745).

## Introduction

### Lightcone creation

We create the lightcones in the following steps, first we submit the soap extraction job

```shell
bash src/make_lightcones.slurm
```

which calls `src/__fast_make_lightcone_mpi.py`, this is the main script that
makes the lightcone and extract needed properties from the
[SOAP](https://arxiv.org/pdf/2507.22669) outputs.

Then we do

```shell
bash src/make_catalogues.slurm
```

This script calls six python scripts, we list them here:

```shell
src/_1_combine_lightcone.py
src/_2_band_patch.py 
src/_3_rotate_lightcone.py
src/1make-samples.py
src/2link-tree-remove-duplicates.py
src/3remove-outliers.py
```

The utilities of each script is briefly described in the head of the file.

After the two steps we would have the cluster catalogue(s) in the format
suitable for our anisotropy analysis. The format is merely `.csv` lists of
clusters with their properties, only that the name of the columns are recognised
with our code.

### H0 analysis

For the H0 anisotropy analysis, we use scripts in the `src/h0` folder.
`batch_h0mc.slurm` and `h0mc.py` are the main scripts. We have 4 other variations:
`h0mc_scatter` for the test with injected scatter; `h0mc_joint` for joint
analysis using two relations; `h0mc_joint_scatter` for the joint analysis using
two relations and with injected scatter; and `h0mc_zcos` for the test that uses
the cosmological redshift with no peculiar velocity and thus no anisotropy
except statistical fluctuation.

See the scipts and the documentation therein for what they do exactly.

### Bulk flow

Similar to the h0 analysis, the scripts for the bulk flow analysis is under the
`src/bf` folder. There is the basic run `bfmc.py` and `batch_bfmc.sh`, two
variants, `bfmc_zcos` with cosmological redshift, and `bfmc_scatter` which uses
the injected scatter.

## Contact

This repository is mainly to keep the work open and reproducible . As a result,
the code might not be as user-friendly, but over time I will gradually put
in things to make it easier to use (which is also important for its number one
user, me)! If you have any questions, comments, or if you spotted a bug, feel
free to drop me an email or open an issue! I'll be happy to answer.

Email: yujiehe@strw.leidenuniv.nl