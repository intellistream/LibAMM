QCD: Quantum pHYSICS
## Background. 
Lattice gauge theory is a discretization of quantum chromodynamics which is generally accepted to be the fundamental physical theory of strong interactions among the quarks as constituents of matter. The most time-consuming part of a numerical simulation in lattice gauge theory with Wilson fermions on the lattice is the computation of quark propagators within a chromodynamic background gauge field. These computations use up a major part of the world's high performance computing power.

Quark propagators are obtained by solving the inhomogeneous lattice Dirac equation Ax = b, where A = I - kD with 0 <= k < kc is a large but sparse complex non-Hermitian matrix representing a periodic nearest-neighbour coupling on a four-dimensional Euclidean space-time lattice.

From the physical theory it is clear that the matrix A should be positive real (all eigenvalues lie in the right half plane) for 0 <= k < kc. Here, kc represents a critical parameter which depends on the given matrix D. Denoting

\gamma_5 = \left( \begin{array}{cccc}0 & 0 & 1 & 0 \\0 & 0 & 0 & 1 \\1 & 0 & 0 & 0 \\0 & 1 & 0 & 0 \end{array}\right)
the Wilson fermion matrix A is Gamma-5 symmetric,
\Gamma_5 A = A^H \Gamma_5,\;\;\;\;\Gamma_5 = I \otimes ( \gamma_5 \otimes I_3 )
Due to the nearest neighbour coupling, the matrix A has 'property A'. This means that with a red-black (or odd-even) ordering of the grid points the matrix becomes

A = I - kD
with
D = \left( \begin{array}{cc}0 & D_{\rm oe}  \\D_{\rm eo} & 0  \end{array}\right)


**Note** We have pre-processed the complex matrixies into their modulus:
- conf5.0-00l4x4-1000 : qcda_small
- conf5.0-00l4x4-2600 : qcdb_small
- conf5.4-00l8x8-2000 : qcda_large
- conf6.0-00l8x8-2000 : qcdb_large
https://math.nist.gov/MatrixMarket/data/misc/qcd/qcd.html


