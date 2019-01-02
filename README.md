# Dragonfly

*Dragonfly* is a software package to analyze single particle diffractive imaging data. The package has the following parts

* An implementation of the [EMC single-particle reconstruction algorithm](http://journals.aps.org/pre/abstract/10.1103/PhysRevE.80.026705) using MPI and OpenMP to merge diffraction patterns of approximately identical particles in random orientations.
* Data stream simulator, that generates noisy single-particle diffraction patterns from a [PDB](http://www.rcsb.org/pdb/home/home.do) file
* An experimental pattern-classification GUI using some machine learning tools to separate single particle diffraction patterns in experimental data.

More detailed documentation can be found in the [Wiki pages](https://github.com/duaneloh/Dragonfly/wiki).

If you are in a hurry, just clone the repository and follow the [Quick start](https://github.com/duaneloh/Dragonfly/wiki/Quick-start-with-simulations) instructions.

Please cite the following publication if you use *Dragonfly* for your work:
> Ayyer, K., Lan, T. Y., Elser, V., & Loh, N. D. (2016). Dragonfly: an implementation of the expand–maximize–compress algorithm for single-particle imaging. [*Journal of applied crystallography*, **49**(4), 1320-1335](https://doi.org/10.1107/S1600576716008165).

