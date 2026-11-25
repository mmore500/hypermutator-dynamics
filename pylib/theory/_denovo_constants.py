import numpy as np

# generations elapsed since current nonmutator nose class was founded, at
# which hold probability is tabulated
HOLD_TAUS = np.array([0, 15, 30, 45, 60, 80, 100, 130, 170, 250, 400])
HOLD_KMAX = 7  # hold is near-certain with more levels than this remaining
HOLD_NOSE = 2  # nonmutator nose level used to tabulate hold probability
MAX_DEL = 4  # genomes with more deleterious mutations than this are lumped
