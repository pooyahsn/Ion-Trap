# Molmer-Sorensen hamiltonian
The hamiltonian is based on the paper: "Quantum Computation with Ions in Thermal Motion" by Molmer and Sorensen(1999). 
One is the total hamiltonian and the other the approximate hamiltonian that is valid in the Lamb-Dicke regime. The parameters are from the paper.

# Native qudit gates
This file implements the native rotations appearing in the following paper: "A universal qudit quantum processor with trapped ions", Ringbauer et al. (2021).(so far only the one qudit rotations)
Arbitrary single qudit gates are first, using givens rotations, decomposed into rotation gates that can be applied physically in their lab.
