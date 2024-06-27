import numpy as np

def compatible_relaxation_rate(e1,e2):
    """
    Params:
    -----------
    e1: vector of cr score from previous iteration
    e2: vector of cr score from current iteration

    Return:
    -----------
    rate: measures the "converge" of cr steps
    """

    return np.linalg.norm(e2)/np.linalg.norm(e1)


