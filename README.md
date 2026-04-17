# Dragonfly

*Dragonfly* is a software package to analyze single particle diffractive imaging data. The package has the following parts:

* An implementation of the [EMC single-particle reconstruction algorithm](http://journals.aps.org/pre/abstract/10.1103/PhysRevE.80.026705) using MPI and OpenMP to merge diffraction patterns of approximately identical particles in random orientations.
* Data stream simulator, that generates noisy single-particle diffraction patterns from a [PDB](http://www.rcsb.org/pdb/home/home.do) file.

## Installation

```
pip install dragonfly-spi
```

## Quick Start

Initialize a reconstruction directory with default configuration files:
```
dragonfly.init
cd recon_0001
```

Edit `config.ini` to set your PDB file, detector geometry and other parameters, then generate simulated data:
```
dragonfly.utils.sim_setup
```

Run the EMC reconstruction (here with 4 OpenMP threads for 10 iterations):
```
dragonfly.emc -t 4 10
```

Monitor progress with the plotting GUI:
```
dragonfly.autoplot
```

More detailed documentation can be found in the [online docs](https://dragonfly-spi.readthedocs.io).

## Citation

Please cite the following publication if you use *Dragonfly* for your work:
> Ayyer, K., Lan, T. Y., Elser, V., & Loh, N. D. (2016). Dragonfly: an implementation of the expand–maximize–compress algorithm for single-particle imaging. [*Journal of applied crystallography*, **49**(4), 1320-1335](https://doi.org/10.1107/S1600576716008165).
