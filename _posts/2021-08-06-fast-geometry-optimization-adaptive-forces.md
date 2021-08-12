---
layout: post
# mathjax: true
title: Atomistic Geometry Optimization With Adaptively Converged Forces
date: 2021-08-06
category:
  - Blog
tags:
  - Python
  - DFT
  - GPAW
  - Optimization
---

Atomistic geometry optimization in _ab initio_ calculations is almost always the first step of a typical theoretical study on materials, at least where I am. The goal is to minimize the magnitude of forces such that the maximum force acting on any atom in the system is below a threshold (typically below 1-10 meV / Angstrom (A from hereon until I figure out the mathjax...). This can take anywhere from a few minutes for a small system (<10-20 atoms) on a desktop workstation, to hours, even days for a large system (>200-300 atoms), even with a few hundred supercomputer cores.

Today we're going to have a look at how we can potentially speed up the optimization of the geometry by telling our force calculator (<a href="https://en.wikipedia.org/wiki/Density_functional_theory" target="_blank">density functional theory</a> code <a href="https://wiki.fysik.dtu.dk/gpaw/" target="_blank">GPAW</a>) to adaptively _only_ converge the density and wavefunctions of our system such that our forces are converged to only a fraction (I'll be using 0.1 here) of the maximum force in the system.

What is the rationale here? Suppose you have a system such that the largest force on any atom is 3 eV/A. If you want to optimize to 1 meV/A, and can only have a single number as your force convergence in your SCF, you'll be converging to 0.1 meV/A from the very first iteration. _There is no need to do this._ In a gradient-based optimization problem, the algorithm that uses gradients and approximate Hessians from these gradients (<a href="https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm" target="_blank">BFGS</a> is the first candidate here that comes to mind) doesn't really care about your gradient being accurate to 5 significant figures. If your maximum component of your gradient is 3 eV/A, it is reasonable to claim that it doesn't really matter if it's 3.0000 eV/A or 3.0001 eV/A. Numerically, your algorithm cares about no more than 1-2 significant figures of your gradient. Any more precision, and the returns rapidly diminish.

To proceed, we will make a custom `RelativeForces` class, which will be an instance of GPAW's `Criterion` class. We can then use this to tell our DFT calculator that our self-consistent field (SCF) convergence is converged when the maximum change in the magnitude of the forces is 1/10th the maximum magnitude of the forces. So if our maximum force magnitude is 3 eV/A, we only converge our SCF till the max change (on any atom, but extremely likely the atom with the highest force on it) is no more than 0.3 eV/A.

You can read more about custom convergence criteria <a href="https://wiki.fysik.dtu.dk/gpaw/documentation/custom_convergence.html" target="_blank">here</a>.

```python
import numpy as np

from ase.units import Ha, Bohr

from gpaw.scf import Criterion
from gpaw.forces import calculate_forces


class RelativeForces(Criterion):
    name = 'rel-forces'
    tablename = 'rel-f'

    def __init__(self, tol, calc_last=True):
        self.tol = tol
        self.description = ('Maximum relative change in the atomic forces across '
                            'last 2 cycles: {:g}'.format(self.tol))
        self.calc_last = calc_last
        self.reset()

    def __call__(self, context):
        """Should return (bool, entry), where bool is True if converged and
        False if not, and entry is a <=5 character string to be printed in
        the user log file."""
        with context.wfs.timer('Forces'):
            F_av = calculate_forces(context.wfs, context.dens, context.ham)
            F_av *= Ha / Bohr
        error = np.inf
        if self.old_F_av is not None:

            error = np.max(np.linalg.norm(F_av - self.old_F_av, axis=1)) / \
                np.max(np.linalg.norm(F_av, axis=1))
        self.old_F_av = F_av
        converged = (error < self.tol)
        entry = ''
        if np.isfinite(error):
            entry = '{:+5.2f}'.format(np.log10(error))
        return converged, entry

    def reset(self):
        self.old_F_av = None
```

The `Criterion` above checks if the forces over the past two SCF iterations have changed by more `tol` times the current forces `F_av` (the name `F_av` stands for 'an array called F, the first index of which is a, atoms, and the second of which is v, vectors'). It's not a name that complies with conventional best practices, but GPAW uses names following this convention since they are quite useful when developing.

Right, now let's set up the rest of the calculator, optimizer, etc:

```python
import numpy as np

from gpaw import GPAW
from gpaw.scf import Criterion
from gpaw.forces import calculate_forces


from ase.build import mx2
from ase.optimize import BFGS
from ase.units import Ha, Bohr

atoms = mx2(vacuum=5) * [2, 2, 1]


class RelativeForces(Criterion):
    name = 'rel-forces'
    tablename = 'relforce'

    def __init__(self, tol, calc_last=True):
        self.tol = tol
        self.description = ('Maximum change in the atomic [forces] across '
                            'last 2 cycles: {:g} eV/Ang'.format(self.tol))
        self.calc_last = calc_last
        self.reset()

    def __call__(self, context):
        """Should return (bool, entry), where bool is True if converged and
        False if not, and entry is a <=5 character string to be printed in
        the user log file."""
        if np.isinf(self.tol):  # criterion is off; backwards compatibility
            return True, ''
        with context.wfs.timer('Forces'):
            F_av = calculate_forces(context.wfs, context.dens, context.ham)
            F_av *= Ha / Bohr
        error = np.inf
        if self.old_F_av is not None:
            error = ((F_av - self.old_F_av)**2).sum(1).max()**0.5 / \
                np.max(np.linalg.norm(F_av, axis=1))
        self.old_F_av = F_av
        converged = (error < self.tol)
        entry = ''
        if np.isfinite(error):
            entry = '{:+5.2f}'.format(np.log10(error))
        return converged, entry

    def reset(self):
        self.old_F_av = None


adaptive_force_convergence = False
if not adaptive_force_convergence:
    convergence = {'forces': 1e-4}
    text_output = 'outbrute.txt'
else:
    convergence = {'eigenstates': 1e-5, 'density': 3e-4,
                   'custom': [RelativeForces(0.1)]},
    text_output = 'outadaptive.txt'

calc = GPAW(mode='pw',
            kpts=[6, 6, 1],
            convergence=convergence,
            txt=text_output
            )

atoms.rattle(0.1)
atoms.calc = calc

opt = BFGS(atoms)
opt.run(fmax=1e-3)
```

We are going to converge our forces till our relative error in the maximum force component is below 0.1. Notice how I've also loosened the `eigenstates` and `density` convergence criteria from their defaults, since otherwise in the few few coarse steps, we'd be converging too tightly anyway. I'm also making a tradeoff, since near the end, the density and eigenstates will converge too early, and a small amount of extra compute time will have to be used to calculate the <a href="https://en.wikipedia.org/wiki/Hellmann%E2%80%93Feynman_theorem" target="_blank">Hellmann-Feynman</a> forces at every SCF iteration. The hope is that this extra time required towards the end is so small that our initial savings, when the forces are still large, far outweigh these costs. Let's have a look at the outputs.

Constant force convergence of 1e-4 eV/A from the beginning:
```
      Step     Time          Energy         fmax
BFGS:    0 21:15:10      -95.873739        6.1800
BFGS:    1 21:16:34      -97.139433        3.4207
BFGS:    2 21:17:55      -97.961585        1.3522
BFGS:    3 21:19:11      -98.065318        1.0545
BFGS:    4 21:20:26      -98.217607        0.8072
BFGS:    5 21:21:32      -98.244415        0.5895
BFGS:    6 21:22:39      -98.289952        0.2996
BFGS:    7 21:23:32      -98.298972        0.2731
BFGS:    8 21:24:15      -98.317347        0.2320
BFGS:    9 21:25:00      -98.320285        0.1837
BFGS:   10 21:25:43      -98.322307        0.1129
BFGS:   11 21:26:31      -98.323191        0.0733
BFGS:   12 21:27:11      -98.323604        0.0320
BFGS:   13 21:27:48      -98.323682        0.0151
BFGS:   14 21:28:22      -98.323705        0.0106
BFGS:   15 21:28:55      -98.323719        0.0075
BFGS:   16 21:29:20      -98.323728        0.0080
BFGS:   17 21:29:45      -98.323734        0.0064
BFGS:   18 21:30:03      -98.323737        0.0042
BFGS:   19 21:30:29      -98.323738        0.0025
BFGS:   20 21:30:54      -98.323739        0.0021
BFGS:   21 21:31:12      -98.323739        0.0016
BFGS:   22 21:31:30      -98.323740        0.0007

```

So this above run takes about 1067 seconds.

Adaptive force convergence of 1e-1:
```
      Step     Time          Energy         fmax
BFGS:    0 20:55:36      -95.873746        6.1763
BFGS:    1 20:56:12      -97.139023        3.4381
BFGS:    2 20:56:51      -97.957347        1.3814
BFGS:    3 20:57:15      -98.062531        1.0355
BFGS:    4 20:57:45      -98.219518        0.7097
BFGS:    5 20:58:07      -98.247846        0.5393
BFGS:    6 20:58:35      -98.285280        0.3471
BFGS:    7 20:58:59      -98.298637        0.2878
BFGS:    8 20:59:22      -98.311309        0.2634
BFGS:    9 20:59:44      -98.318648        0.2127
BFGS:   10 21:00:07      -98.321592        0.1438
BFGS:   11 21:00:32      -98.323015        0.0895
BFGS:   12 21:00:52      -98.323368        0.0534
BFGS:   13 21:01:20      -98.323656        0.0271
BFGS:   14 21:01:49      -98.323693        0.0137
BFGS:   15 21:02:15      -98.323708        0.0102
BFGS:   16 21:02:39      -98.323719        0.0078
BFGS:   17 21:03:12      -98.323732        0.0056
BFGS:   18 21:03:36      -98.323736        0.0039
BFGS:   19 21:03:59      -98.323738        0.0028
BFGS:   20 21:04:29      -98.323739        0.0020
BFGS:   21 21:04:52      -98.323739        0.0016
BFGS:   22 21:05:18      -98.323739        0.0010
```
The run _now_ takes 635 seconds. We have saved ~40% in calculation time!

The final energies are equal up to tolerance, so the two approaches get the same answer.

Here is my `gpaw info` output:
```
 -----------------------------------------------------------------------------------------------------
| python-3.8.10             /home/chronum/miniconda3/envs/gpawhack/bin/python                         |
| gpaw-21.6.1b1-e48871cb1a  /home/chronum/GPAW/gpawhacksrc/gpaw/                                      |
| ase-3.22.0                /home/chronum/miniconda3/envs/gpawhack/lib/python3.8/site-packages/ase/   |
| numpy-1.21.1              /home/chronum/miniconda3/envs/gpawhack/lib/python3.8/site-packages/numpy/ |
| scipy-1.7.0               /home/chronum/miniconda3/envs/gpawhack/lib/python3.8/site-packages/scipy/ |
| libxc-4.3.4               yes                                                                       |
| _gpaw-5d5b0fc3a6          /home/chronum/GPAW/gpawhacksrc/_gpaw.cpython-38-x86_64-linux-gnu.so       |
| MPI enabled               yes                                                                       |
| OpenMP enabled            no                                                                        |
| scalapack                 yes                                                                       |
| Elpa                      no                                                                        |
| FFTW                      yes                                                                       |
| libvdwxc                  no                                                                        |
| PAW-datasets (1)          /home/chronum/GPAW/setups/gpaw-basis-pvalence-0.9.20000                   |
| PAW-datasets (2)          /home/chronum/GPAW/setups/gpaw-setups-0.9.20000                           |
 -----------------------------------------------------------------------------------------------------
```

All calculations were run on a Fedora 34 virtual machine running inside VirtualBox with 8 threads assigned to it. The CPU is a Ryzen 7 5800X processor with 8 cores.

In conclusion:

We (presumably you the reader, and I) have just gone through a way to accelerate atomistic geometry optimization runs using adaptive convergence of the forces during the SCF step. We have (hopefully) taken note of the modular design of `Criterion` objects that allow the user to supply effectively any meaningful criterion that can act as a desired variable to be converged during a full SCF convergence. This will hopefully act as a guide, and as inspiration for other users/developers of GPAW, as well as other DFT codes to enable flexibility of this kind that ultimately leads to extremely nontrivial increases in computational efficiency.
