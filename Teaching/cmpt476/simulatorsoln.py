import numpy as np
from numpy import random

# ----------------------------------------
# Utilities
# ----------------------------------------

def apply_gate(gate, state):
    """
    Apply a quantum gate to a state vector
    """
    return gate @ state
    

def toBits(i, n):
    """
    Expand an integer i to its binary n-bit representation (Big-endian)
    """
    if i < 0:
    	i = i % (1 << n)

    return [(i >> k) & 1 for k in range(n - 1, -1, -1)]

# ----------------------------------------
# Q1: Quantum states
# ----------------------------------------

KET_0 = np.array([[1], [0]])
KET_1 = np.array([[0], [1]])


# ----------------------------------------
# Q2: Quantum gates
# ----------------------------------------

I_GATE = np.array([[1,0],[0,1]], dtype=complex)
X_GATE = np.array([[0,1],[1,0]], dtype=complex)
Y_GATE = np.array([[0,1j],[-1j,0]], dtype=complex)
Z_GATE = np.array([[1,0],[0,-1]], dtype=complex)
H_GATE = 1/np.sqrt(2)*np.array([[1,1],[1,-1]], dtype=complex)

CNOT_GATE = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)

# ----------------------------------------
# Q3: Implementing utilities
# ----------------------------------------

def normalize(state):
    """
    Normalize a state vector.
    """
    sum = 0
    for r in state:
        sum += r[0] * r[0].conjugate()

    return state / np.sqrt(sum)

def tensor(a, b):
    """
    Tensor product of two state vectors or operators
    """

    res = []
    for s in a:
        for t in b:
            row = []
            for x in s:
                for y in t: 
                    row.append(x*y) 
            res.append(row)

    return np.array(res)

# ----------------------------------------
# Q4: Implementing measurement
# ----------------------------------------
   

def measure(state, i, n):
    """
    Measure a quantum state in the computational basis.

    Returns:
        outcome (int)
        collapsed_state (numpy array)
        
    e.g.
    	measure(KET_0) == (0, KET_0)
    	measure(KET_1) == (1, KET_1)
    """
    
    # Split the state into two marginal states
    ket0, ket1 = [], []
    for j in range(len(state)):
        bits = toBits(j, n)
        if bits[i] == 0:
            ket0.append([state[j][0]])
            ket1.append([0])
        else:
            ket0.append([0])
            ket1.append([state[j][0]])

    # Compute the probabilities
    p0 = sum(z[0]*z[0].conjugate() for z in ket0)
    p1 = sum(z[0]*z[0].conjugate() for z in ket1)

    # Sample from the distribution
    result = random.choice([0,1], p=[p0,p1])
    if result == 0:
        return (0,ket0/np.sqrt(p0))
    else:
        return (1,ket1/np.sqrt(p1))


# ----------------------------------------
# Tests
# ----------------------------------------

def test_hadamard_on_zero():
    """
    H |0> = |+>
    """
    lhs = apply_gate(H_GATE, KET_0)
    rhs = (KET_0 + KET_1) / np.sqrt(2)

    assert np.allclose(lhs, rhs), "H|0> != |+>"
    print("Test passed: H|0> = |+>")


def test_bell_state():
    """
    CNOT (H tensor I) |00> = 1/2(|00> + |11>)
    """
    lhs = CNOT_GATE @ tensor(H_GATE, I_GATE) @ tensor(KET_0, KET_0)
    rhs = (tensor(KET_0, KET_0) + tensor(KET_1, KET_1)) / np.sqrt(2)

    assert np.allclose(lhs, rhs), "CNOT (H tensor I) |00> != 1/2(|00> + |11>)"
    print("Test passed: CNOT (H tensor I) |00> = 1/2(|00> + |11>)")


def test_hxh_equals_z():
    """
    H X H = Z
    """
    lhs = H_GATE @ X_GATE @ H_GATE
    rhs = Z_GATE

    assert np.allclose(lhs, rhs), "HXH != Z"
    print("Test passed: HXH = Z")

def test_h_tensor_x():
    """
    H tensor X
    """
    lhs = tensor(H_GATE, X_GATE)
    rhs = 1/np.sqrt(2)*np.array([[0,1,0,1],[1,0,1,0],[0,1,0,-1],[1,0,-1,0]])

    assert np.allclose(lhs, rhs), "H tensor X incorrect"
    print("Test passed: H tensor X")

def test_measure():
    """
    Measurement test
    """
    psi = (1/np.sqrt(2)*tensor(KET_0,KET_0) + 
           1/np.sqrt(3)*tensor(KET_1,KET_0) -
           1/np.sqrt(6)*tensor(KET_1,KET_1))

    psiA0 = tensor(KET_0,KET_0)
    psiA1 = np.sqrt(2)/np.sqrt(3)*tensor(KET_1,KET_0) - 1/np.sqrt(3)*tensor(KET_1,KET_1)
    psiB0 = np.sqrt(3)/np.sqrt(5)*tensor(KET_0,KET_0) + np.sqrt(2)/np.sqrt(5)*tensor(KET_1,KET_0)
    psiB1 = -tensor(KET_1,KET_1)

    resultsA = [psiA0,psiA1]
    resultsB = [psiB0,psiB1]

    for i in range(10):
        res,state = measure(psi, 0, 2)

        assert np.allclose(resultsA[res], state), "Measurement error"

    for i in range(10):
        res,state = measure(psi, 1, 2)

        assert np.allclose(resultsB[res], state), "Measurement error"

    print("Test passed: Measurement")


if __name__ == "__main__":

    # Run tests
    test_hadamard_on_zero()
    test_bell_state()
    test_hxh_equals_z()
    test_h_tensor_x()
    test_measure()

