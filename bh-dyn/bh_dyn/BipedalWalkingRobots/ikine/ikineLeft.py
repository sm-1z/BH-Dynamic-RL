import numpy as np

def forward_kinematics_Left(q, L, D, Phi):
    T_current = np.zeros((4,4))

    T_current[0, 0] = np.cos(q[0]) * np.cos(q[1]) * np.sin(q[5]) - np.cos(Phi[0] + Phi[1] + Phi[2] - q[4]) * np.cos(q[2]) * np.cos(q[5]) * np.sin(q[0]) + np.sin(Phi[0] + Phi[1] + Phi[2] - q[4]) * np.cos(q[5]) * np.sin(q[0]) * np.sin(q[2]) - np.cos(Phi[0] + Phi[1] + Phi[2] - q[4]) * np.cos(q[0]) * np.cos(q[5]) * np.sin(q[1]) * np.sin(q[2]) - np.sin(Phi[0] + Phi[1] + Phi[2] - q[4]) * np.cos(q[0]) * np.cos(q[2]) * np.cos(q[5]) * np.sin(q[1])
    T_current[0, 1] = np.cos(q[0]) * np.cos(q[1]) * np.cos(q[5]) + np.cos(Phi[0] + Phi[1] + Phi[2] - q[4]) * np.cos(q[2]) * np.sin(q[0]) * np.sin(q[5]) - np.sin(Phi[0] + Phi[1] + Phi[2] - q[4]) * np.sin(q[0]) * np.sin(q[2]) * np.sin(q[5]) + np.cos(Phi[0] + Phi[1] + Phi[2] - q[4]) * np.cos(q[0]) * np.sin(q[1]) * np.sin(q[2]) * np.sin(q[5]) + np.sin(Phi[0] + Phi[1] + Phi[2] - q[4]) * np.cos(q[0]) * np.cos(q[2]) * np.sin(q[1]) * np.sin(q[5])
    T_current[0, 2] = np.cos(Phi[0] + Phi[1] + Phi[2] - q[4]) * np.sin(q[0]) * np.sin(q[2]) + np.sin(Phi[0] + Phi[1] + Phi[2] - q[4]) * np.cos(q[2]) * np.sin(q[0]) - np.cos(Phi[0] + Phi[1] + Phi[2] - q[4]) * np.cos(q[0]) * np.cos(q[2]) * np.sin(q[1]) + np.sin(Phi[0] + Phi[1] + Phi[2] - q[4]) * np.cos(q[0]) * np.sin(q[1]) * np.sin(q[2])
    T_current[1, 0] = -np.sin(q[1]) * np.sin(q[5]) - np.cos(Phi[0] + Phi[1] + Phi[2] - q[4]) * np.cos(q[1]) * np.cos(q[5]) * np.sin(q[2]) - np.sin(Phi[0] + Phi[1] + Phi[2] - q[4]) * np.cos(q[1]) * np.cos(q[2]) * np.cos(q[5])
    T_current[1, 1] = np.cos(Phi[0] + Phi[1] + Phi[2] - q[4]) * np.cos(q[1]) * np.sin(q[2]) * np.sin(q[5]) - np.cos(q[5]) * np.sin(q[1]) + np.sin(Phi[0] + Phi[1] + Phi[2] - q[4]) * np.cos(q[1]) * np.cos(q[2]) * np.sin(q[5])
    T_current[1, 2] = -np.cos(Phi[0] + Phi[1] + Phi[2] + q[2] - q[4]) * np.cos(q[1])
    T_current[2, 0] = np.sin(Phi[0] + Phi[1] + Phi[2] - q[4]) * np.cos(q[0]) * np.cos(q[5]) * np.sin(q[2]) - np.cos(q[1]) * np.sin(q[0]) * np.sin(q[5]) - np.cos(Phi[0] + Phi[1] + Phi[2] - q[4]) * np.cos(q[0]) * np.cos(q[2]) * np.cos(q[5]) + np.cos(Phi[0] + Phi[1] + Phi[2] - q[4]) * np.cos(q[5]) * np.sin(q[0]) * np.sin(q[1]) * np.sin(q[2]) + np.sin(Phi[0] + Phi[1] + Phi[2] - q[4]) * np.cos(q[2]) * np.cos(q[5]) * np.sin(q[0]) * np.sin(q[1])
    T_current[2, 1] = np.cos(Phi[0] + Phi[1] + Phi[2] - q[4]) * np.cos(q[0]) * np.cos(q[2]) * np.sin(q[5]) - np.cos(q[1]) * np.cos(q[5]) * np.sin(q[0]) - np.sin(Phi[0] + Phi[1] + Phi[2] - q[4]) * np.cos(q[0]) * np.sin(q[2]) * np.sin(q[5]) - np.cos(Phi[0] + Phi[1] + Phi[2] - q[4]) * np.sin(q[0]) * np.sin(q[1]) * np.sin(q[2]) * np.sin(q[5]) - np.sin(Phi[0] + Phi[1] + Phi[2] - q[4]) * np.cos(q[2]) * np.sin(q[0]) * np.sin(q[1]) * np.sin(q[5])
    T_current[2, 2] = np.cos(Phi[0] + Phi[1] + Phi[2] - q[4]) * np.cos(q[0]) * np.sin(q[2]) + np.sin(Phi[0] + Phi[1] + Phi[2] - q[4]) * np.cos(q[0]) * np.cos(q[2]) + np.cos(Phi[0] + Phi[1] + Phi[2] - q[4]) * np.cos(q[2]) * np.sin(q[0]) * np.sin(q[1]) - np.sin(Phi[0] + Phi[1] + Phi[2] - q[4]) * np.sin(q[0]) * np.sin(q[1]) * np.sin(q[2])

    T_current[0, 3] = L[1] * (np.sin(q[0]) * np.sin(q[2]) - np.cos(q[0]) * np.cos(q[2]) * np.sin(q[1])) - (np.cos(q[3]) * (np.cos(q[2]) * np.sin(q[0]) + np.cos(q[0]) * np.sin(q[1]) * np.sin(q[2])) + np.sin(q[3]) * (np.sin(q[0]) * np.sin(q[2]) - np.cos(q[0]) * np.cos(q[2]) * np.sin(q[1]))) * (L[2] + L[4] * np.cos(Phi[0] + Phi[1] + q[3]) + L[3] * np.cos(Phi[0])) - L[5] * (np.cos(Phi[0] + Phi[1] + Phi[2] - q[4]) * np.cos(q[2]) * np.cos(q[5]) * np.sin(q[0]) - np.cos(q[0]) * np.cos(q[1]) * np.sin(q[5]) - np.sin(Phi[0] + Phi[1] + Phi[2] - q[4]) * np.cos(q[5]) * np.sin(q[0]) * np.sin(q[2]) + np.cos(Phi[0] + Phi[1] + Phi[2] - q[4]) * np.cos(q[0]) * np.cos(q[5]) * np.sin(q[1]) * np.sin(q[2]) + np.sin(Phi[0] + Phi[1] + Phi[2] - q[4]) * np.cos(q[0]) * np.cos(q[2]) * np.cos(q[5]) * np.sin(q[1])) - L[0] + (np.cos(q[3]) * (np.sin(q[0]) * np.sin(q[2]) - np.cos(q[0]) * np.cos(q[2]) * np.sin(q[1])) - np.sin(q[3]) * (np.cos(q[2]) * np.sin(q[0]) + np.cos(q[0]) * np.sin(q[1]) * np.sin(q[2]))) * (L[4] * np.sin(Phi[0] + Phi[1] + q[3]) + L[3] * np.sin(Phi[0])) - D[0] * np.sin(q[0]) - D[1] * np.cos(q[0]) * np.cos(q[1])
    T_current[1, 3] = D[0] - L[5] * (np.sin(q[1]) * np.sin(q[5]) + np.cos(Phi[0] + Phi[1] + Phi[2] - q[4]) * np.cos(q[1]) * np.cos(q[5]) * np.sin(q[2]) + np.sin(Phi[0] + Phi[1] + Phi[2] - q[4]) * np.cos(q[1]) * np.cos(q[2]) * np.cos(q[5])) + D[1] * np.sin(q[1]) - np.cos(q[2] - q[3]) * np.cos(q[1]) * (L[4] * np.sin(Phi[0] + Phi[1] + q[3]) + L[3] * np.sin(Phi[0])) - L[1] * np.cos(q[1]) * np.cos(q[2]) - np.sin(q[2] - q[3]) * np.cos(q[1]) * (L[2] + L[4] * np.cos(Phi[0] + Phi[1] + q[3]) + L[3] * np.cos(Phi[0]))
    T_current[2, 3] = L[1] * (np.cos(q[0]) * np.sin(q[2]) + np.cos(q[2]) * np.sin(q[0]) * np.sin(q[1])) - (np.cos(q[3]) * (np.cos(q[0]) * np.cos(q[2]) - np.sin(q[0]) * np.sin(q[1]) * np.sin(q[2])) + np.sin(q[3]) * (np.cos(q[0]) * np.sin(q[2]) + np.cos(q[2]) * np.sin(q[0]) * np.sin(q[1]))) * (L[2] + L[4] * np.cos(Phi[0] + Phi[1] + q[3]) + L[3] * np.cos(Phi[0])) + (np.cos(q[3]) * (np.cos(q[0]) * np.sin(q[2]) + np.cos(q[2]) * np.sin(q[0]) * np.sin(q[1])) - np.sin(q[3]) * (np.cos(q[0]) * np.cos(q[2]) - np.sin(q[0]) * np.sin(q[1]) * np.sin(q[2]))) * (L[4] * np.sin(Phi[0] + Phi[1] + q[3]) + L[3] * np.sin(Phi[0])) - D[0] * np.cos(q[0]) + L[5] * (np.cos(q[5]) * (np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * (np.cos(q[3]) * (np.cos(q[0]) * np.sin(q[2]) + np.cos(q[2]) * np.sin(q[0]) * np.sin(q[1])) - np.sin(q[3]) * (np.cos(q[0]) * np.cos(q[2]) - np.sin(q[0]) * np.sin(q[1]) * np.sin(q[2]))) - np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * (np.cos(q[3]) * (np.cos(q[0]) * np.cos(q[2]) - np.sin(q[0]) * np.sin(q[1]) * np.sin(q[2])) + np.sin(q[3]) * (np.cos(q[0]) * np.sin(q[2]) + np.cos(q[2]) * np.sin(q[0]) * np.sin(q[1])))) - np.cos(q[1]) * np.sin(q[0]) * np.sin(q[5])) + D[1] * np.cos(q[1]) * np.sin(q[0])

    T_current[3, 0] = 0
    T_current[3, 1] = 0
    T_current[3, 2] = 0
    T_current[3, 3] = 1

    return T_current

def compute_jacobian_Left(q, L, D, Phi):
    J = np.zeros((6, 6))

    J[0, 0] = -(np.cos(q[1]) * np.sin(q[5]) + np.sin(q[1]) * (np.cos(q[2]) * (np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5]) * np.sin(q[3]) - np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[3]) * np.cos(q[5])) - np.sin(q[2]) * (np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[3]) * np.cos(q[5]) + np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5]) * np.sin(q[3])))) * (D[0] + np.cos(q[2]) * (np.sin(q[3]) * (L[4] * np.sin(Phi[0] + Phi[1] + q[3]) + L[3] * np.sin(Phi[0]) + L[5] * np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5])) + np.cos(q[3]) * (L[2] + L[4] * np.cos(Phi[0] + Phi[1] + q[3]) + L[3] * np.cos(Phi[0]) + L[5] * np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5]))) - np.sin(q[2]) * (L[1] - np.sin(q[3]) * (L[2] + L[4] * np.cos(Phi[0] + Phi[1] + q[3]) + L[3] * np.cos(Phi[0]) + L[5] * np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5])) + np.cos(q[3]) * (L[4] * np.sin(Phi[0] + Phi[1] + q[3]) + L[3] * np.sin(Phi[0]) + L[5] * np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5])))) - (np.cos(q[1]) * (D[1] - L[5] * np.sin(q[5])) + np.sin(q[1]) * (np.sin(q[2]) * (np.sin(q[3]) * (L[4] * np.sin(Phi[0] + Phi[1] + q[3]) + L[3] * np.sin(Phi[0]) + L[5] * np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5])) + np.cos(q[3]) * (L[2] + L[4] * np.cos(Phi[0] + Phi[1] + q[3]) + L[3] * np.cos(Phi[0]) + L[5] * np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5]))) + np.cos(q[2]) * (L[1] - np.sin(q[3]) * (L[2] + L[4] * np.cos(Phi[0] + Phi[1] + q[3]) + L[3] * np.cos(Phi[0]) + L[5] * np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5])) + np.cos(q[3]) * (L[4] * np.sin(Phi[0] + Phi[1] + q[3]) + L[3] * np.sin(Phi[0]) + L[5] * np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5]))))) * (np.cos(q[2]) * (np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[3]) * np.cos(q[5]) + np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5]) * np.sin(q[3])) + np.sin(q[2]) * (np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5]) * np.sin(q[3]) - np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[3]) * np.cos(q[5])))
    J[0, 1] = (np.cos(q[2]) * (np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5]) * np.sin(q[3]) - np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[3]) * np.cos(q[5])) - np.sin(q[2]) * (np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[3]) * np.cos(q[5]) + np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5]) * np.sin(q[3]))) * (D[1] - L[5] * np.sin(q[5])) - np.sin(q[5]) * (np.sin(q[2]) * (np.sin(q[3]) * (L[4] * np.sin(Phi[0] + Phi[1] + q[3]) + L[3] * np.sin(Phi[0]) + L[5] * np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5])) + np.cos(q[3]) * (L[2] + L[4] * np.cos(Phi[0] + Phi[1] + q[3]) + L[3] * np.cos(Phi[0]) + L[5] * np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5]))) + np.cos(q[2]) * (L[1] - np.sin(q[3]) * (L[2] + L[4] * np.cos(Phi[0] + Phi[1] + q[3]) + L[3] * np.cos(Phi[0]) + L[5] * np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5])) + np.cos(q[3]) * (L[4] * np.sin(Phi[0] + Phi[1] + q[3]) + L[3] * np.sin(Phi[0]) + L[5] * np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5]))))
    J[0, 2] = -(np.sin(q[3]) * (L[4] * np.sin(Phi[0] + Phi[1] + q[3]) + L[3] * np.sin(Phi[0]) + L[5] * np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5])) + np.cos(q[3]) * (L[2] + L[4] * np.cos(Phi[0] + Phi[1] + q[3]) + L[3] * np.cos(Phi[0]) + L[5] * np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5]))) * (np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5]) * np.sin(q[3]) - np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[3]) * np.cos(q[5])) - (np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[3]) * np.cos(q[5]) + np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5]) * np.sin(q[3])) * (L[1] - np.sin(q[3]) * (L[2] + L[4] * np.cos(Phi[0] + Phi[1] + q[3]) + L[3] * np.cos(Phi[0]) + L[5] * np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5])) + np.cos(q[3]) * (L[4] * np.sin(Phi[0] + Phi[1] + q[3]) + L[3] * np.sin(Phi[0]) + L[5] * np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5])))
    J[0, 3] = np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5]) * (L[4] * np.sin(Phi[0] + Phi[1] + q[3]) + L[3] * np.sin(Phi[0]) + L[5] * np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5])) - np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5]) * (L[2] + L[4] * np.cos(Phi[0] + Phi[1] + q[3]) + L[3] * np.cos(Phi[0]) + L[5] * np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5]))
    J[0, 4] = 0
    J[0, 5] = 0

    J[1, 0] = (np.cos(q[1]) * (D[1] - L[5] * np.sin(q[5])) + np.sin(q[1]) * (np.sin(q[2]) * (np.sin(q[3]) * (L[4] * np.sin(Phi[0] + Phi[1] + q[3]) + L[3] * np.sin(Phi[0]) + L[5] * np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5])) + np.cos(q[3]) * (L[2] + L[4] * np.cos(Phi[0] + Phi[1] + q[3]) + L[3] * np.cos(Phi[0]) + L[5] * np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5]))) + np.cos(q[2]) * (L[1] - np.sin(q[3]) * (L[2] + L[4] * np.cos(Phi[0] + Phi[1] + q[3]) + L[3] * np.cos(Phi[0]) + L[5] * np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5])) + np.cos(q[3]) * (L[4] * np.sin(Phi[0] + Phi[1] + q[3]) + L[3] * np.sin(Phi[0]) + L[5] * np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5]))))) * (np.cos(q[2]) * (np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[3]) * np.sin(q[5]) + np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.sin(q[3]) * np.sin(q[5])) + np.sin(q[2]) * (np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.sin(q[3]) * np.sin(q[5]) - np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[3]) * np.sin(q[5]))) - (np.cos(q[1]) * np.cos(q[5]) - np.sin(q[1]) * (np.cos(q[2]) * (np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.sin(q[3]) * np.sin(q[5]) - np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[3]) * np.sin(q[5])) - np.sin(q[2]) * (np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[3]) * np.sin(q[5]) + np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.sin(q[3]) * np.sin(q[5])))) * (D[0] + np.cos(q[2]) * (np.sin(q[3]) * (L[4] * np.sin(Phi[0] + Phi[1] + q[3]) + L[3] * np.sin(Phi[0]) + L[5] * np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5])) + np.cos(q[3]) * (L[2] + L[4] * np.cos(Phi[0] + Phi[1] + q[3]) + L[3] * np.cos(Phi[0]) + L[5] * np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5]))) - np.sin(q[2]) * (L[1] - np.sin(q[3]) * (L[2] + L[4] * np.cos(Phi[0] + Phi[1] + q[3]) + L[3] * np.cos(Phi[0]) + L[5] * np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5])) + np.cos(q[3]) * (L[4] * np.sin(Phi[0] + Phi[1] + q[3]) + L[3] * np.sin(Phi[0]) + L[5] * np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5]))))
    J[1, 1] = -(np.cos(q[2]) * (np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.sin(q[3]) * np.sin(q[5]) - np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[3]) * np.sin(q[5])) - np.sin(q[2]) * (np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[3]) * np.sin(q[5]) + np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.sin(q[3]) * np.sin(q[5]))) * (D[1] - L[5] * np.sin(q[5])) - np.cos(q[5]) * (np.sin(q[2]) * (np.sin(q[3]) * (L[4] * np.sin(Phi[0] + Phi[1] + q[3]) + L[3] * np.sin(Phi[0]) + L[5] * np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5])) + np.cos(q[3]) * (L[2] + L[4] * np.cos(Phi[0] + Phi[1] + q[3]) + L[3] * np.cos(Phi[0]) + L[5] * np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5]))) + np.cos(q[2]) * (L[1] - np.sin(q[3]) * (L[2] + L[4] * np.cos(Phi[0] + Phi[1] + q[3]) + L[3] * np.cos(Phi[0]) + L[5] * np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5])) + np.cos(q[3]) * (L[4] * np.sin(Phi[0] + Phi[1] + q[3]) + L[3] * np.sin(Phi[0]) + L[5] * np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5]))))
    J[1, 2] = (np.sin(q[3]) * (L[4] * np.sin(Phi[0] + Phi[1] + q[3]) + L[3] * np.sin(Phi[0]) + L[5] * np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5])) + np.cos(q[3]) * (L[2] + L[4] * np.cos(Phi[0] + Phi[1] + q[3]) + L[3] * np.cos(Phi[0]) + L[5] * np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5]))) * (np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.sin(q[3]) * np.sin(q[5]) - np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[3]) * np.sin(q[5])) + (np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[3]) * np.sin(q[5]) + np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.sin(q[3]) * np.sin(q[5])) * (L[1] - np.sin(q[3]) * (L[2] + L[4] * np.cos(Phi[0] + Phi[1] + q[3]) + L[3] * np.cos(Phi[0]) + L[5] * np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5])) + np.cos(q[3]) * (L[4] * np.sin(Phi[0] + Phi[1] + q[3]) + L[3] * np.sin(Phi[0]) + L[5] * np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5])))
    J[1, 3] = np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.sin(q[5]) * (L[2] + L[4] * np.cos(Phi[0] + Phi[1] + q[3]) + L[3] * np.cos(Phi[0]) + L[5] * np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5])) - np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.sin(q[5]) * (L[4] * np.sin(Phi[0] + Phi[1] + q[3]) + L[3] * np.sin(Phi[0]) + L[5] * np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5]))
    J[1, 4] = 0
    J[1, 5] = L[5]

    J[2, 0] = np.sin(q[1]) * (np.cos(q[2]) * (np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[3]) + np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.sin(q[3])) + np.sin(q[2]) * (np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.sin(q[3]) - np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[3]))) * (D[0] + np.cos(q[2]) * (np.sin(q[3]) * (L[4] * np.sin(Phi[0] + Phi[1] + q[3]) + L[3] * np.sin(Phi[0]) + L[5] * np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5])) + np.cos(q[3]) * (L[2] + L[4] * np.cos(Phi[0] + Phi[1] + q[3]) + L[3] * np.cos(Phi[0]) + L[5] * np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5]))) - np.sin(q[2]) * (L[1] - np.sin(q[3]) * (L[2] + L[4] * np.cos(Phi[0] + Phi[1] + q[3]) + L[3] * np.cos(Phi[0]) + L[5] * np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5])) + np.cos(q[3]) * (L[4] * np.sin(Phi[0] + Phi[1] + q[3]) + L[3] * np.sin(Phi[0]) + L[5] * np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5])))) - (np.cos(q[1]) * (D[1] - L[5] * np.sin(q[5])) + np.sin(q[1]) * (np.sin(q[2]) * (np.sin(q[3]) * (L[4] * np.sin(Phi[0] + Phi[1] + q[3]) + L[3] * np.sin(Phi[0]) + L[5] * np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5])) + np.cos(q[3]) * (L[2] + L[4] * np.cos(Phi[0] + Phi[1] + q[3]) + L[3] * np.cos(Phi[0]) + L[5] * np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5]))) + np.cos(q[2]) * (L[1] - np.sin(q[3]) * (L[2] + L[4] * np.cos(Phi[0] + Phi[1] + q[3]) + L[3] * np.cos(Phi[0]) + L[5] * np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5])) + np.cos(q[3]) * (L[4] * np.sin(Phi[0] + Phi[1] + q[3]) + L[3] * np.sin(Phi[0]) + L[5] * np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5]))))) * (np.cos(q[2]) * (np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.sin(q[3]) - np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[3])) - np.sin(q[2]) * (np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[3]) + np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.sin(q[3])))
    J[2, 1] = -(D[1] - L[5] * np.sin(q[5])) * (np.cos(q[2]) * (np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[3]) + np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.sin(q[3])) + np.sin(q[2]) * (np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.sin(q[3]) - np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[3])))
    J[2, 2] = (np.sin(q[3]) * (L[4] * np.sin(Phi[0] + Phi[1] + q[3]) + L[3] * np.sin(Phi[0]) + L[5] * np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5])) + np.cos(q[3]) * (L[2] + L[4] * np.cos(Phi[0] + Phi[1] + q[3]) + L[3] * np.cos(Phi[0]) + L[5] * np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5]))) * (np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[3]) + np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.sin(q[3])) - (np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.sin(q[3]) - np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[3])) * (L[1] - np.sin(q[3]) * (L[2] + L[4] * np.cos(Phi[0] + Phi[1] + q[3]) + L[3] * np.cos(Phi[0]) + L[5] * np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5])) + np.cos(q[3]) * (L[4] * np.sin(Phi[0] + Phi[1] + q[3]) + L[3] * np.sin(Phi[0]) + L[5] * np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5])))
    J[2, 3] = -np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * (L[4] * np.sin(Phi[0] + Phi[1] + q[3]) + L[3] * np.sin(Phi[0]) + L[5] * np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5])) - np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * (L[2] + L[4] * np.cos(Phi[0] + Phi[1] + q[3]) + L[3] * np.cos(Phi[0]) + L[5] * np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) * np.cos(q[5]))
    J[2, 4] = -L[5] * np.cos(q[5])
    J[2, 5] = 0

    J[3, 0] = -np.cos(q[1]) * (np.cos(q[2]) * (np.cos(q[3]) * np.cos(q[5]) * np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) - np.cos(q[5]) * np.sin(q[3]) * np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4])) + np.sin(q[2]) * (np.cos(q[3]) * np.cos(q[5]) * np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) + np.cos(q[5]) * np.sin(q[3]) * np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]))) - np.sin(q[1]) * np.sin(q[5])
    J[3, 1] = np.cos(q[2]) * (np.cos(q[3]) * np.cos(q[5]) * np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) + np.cos(q[5]) * np.sin(q[3]) * np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4])) - np.sin(q[2]) * (np.cos(q[3]) * np.cos(q[5]) * np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) - np.cos(q[5]) * np.sin(q[3]) * np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]))
    J[3, 2] = -np.sin(q[5])
    J[3, 3] = np.sin(q[5])
    J[3, 4] = np.sin(q[5])
    J[3, 5] = 0

    J[4, 0] = np.cos(q[1]) * (np.cos(q[2]) * (np.cos(q[3]) * np.sin(q[5]) * np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) - np.sin(q[3]) * np.sin(q[5]) * np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4])) + np.sin(q[2]) * (np.cos(q[3]) * np.sin(q[5]) * np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) + np.sin(q[3]) * np.sin(q[5]) * np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]))) - np.cos(q[5]) * np.sin(q[1])
    J[4, 1] = np.sin(q[2]) * (np.cos(q[3]) * np.sin(q[5]) * np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) - np.sin(q[3]) * np.sin(q[5]) * np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4])) - np.cos(q[2]) * (np.cos(q[3]) * np.sin(q[5]) * np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) + np.sin(q[3]) * np.sin(q[5]) * np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]))
    J[4, 2] = -np.cos(q[5])
    J[4, 3] = np.cos(q[5])
    J[4, 4] = np.cos(q[5])
    J[4, 5] = 0

    J[5, 0] = -np.cos(q[1]) * (np.cos(q[2]) * (np.cos(q[3]) * np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) + np.sin(q[3]) * np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4])) - np.sin(q[2]) * (np.cos(q[3]) * np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) - np.sin(q[3]) * np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4])))
    J[5, 1] = -np.cos(q[2]) * (np.cos(q[3]) * np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) - np.sin(q[3]) * np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4])) - np.sin(q[2]) * (np.cos(q[3]) * np.cos(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]) + np.sin(q[3]) * np.sin(Phi[0] + Phi[1] + Phi[2] + q[3] - q[4]))
    J[5, 2] = 0
    J[5, 3] = 0
    J[5, 4] = 0
    J[5, 5] = 1

    return J


def left_leg_inv(L_inagle,  L_inp):
    # 定义 pi
    pi = np.pi

    L0_inagle, L1_inagle, L2_inagle =[L_inagle[i] for i in range(3)]
    L0_inp, L1_inp, L2_inp = [L_inp[i] for i in range(3)]
    
    # 计算期望转换矩阵
    T_desired = np.array([[np.cos(L1_inagle) * np.cos(L2_inagle), np.sin(L0_inagle) * np.sin(L1_inagle) * np.cos(L2_inagle) - np.cos(L0_inagle) * np.sin(L2_inagle), np.cos(L0_inagle) * np.sin(L1_inagle) * np.cos(L2_inagle) + np.sin(L0_inagle) * np.sin(L2_inagle), L0_inp],
                          [np.cos(L1_inagle) * np.sin(L2_inagle), np.sin(L0_inagle) * np.sin(L1_inagle) * np.sin(L2_inagle) + np.cos(L0_inagle) * np.cos(L2_inagle), np.cos(L0_inagle) * np.sin(L1_inagle) * np.sin(L2_inagle) - np.sin(L0_inagle) * np.cos(L2_inagle), L1_inp],
                          [-np.sin(L1_inagle), np.sin(L0_inagle) * np.cos(L1_inagle), np.cos(L0_inagle) * np.cos(L1_inagle), L2_inp],
                          [0, 0, 0, 1]])

    # 初始化参数
    L = np.array([0.25, 0.125, 0.087, 0.42125, 0.45675, 0.03])
    D = np.array([0.141, -0.0225])
    Phi = np.array([-35.96 / 180 * pi, 121.83 / 180 * pi, 5.02 / 180 * pi])
    Theta0 = np.array([0 / 180 * pi, 0 / 180 * pi, 50.3 / 180 * pi, 37.3 / 180 * pi, -40.57 / 180 * pi, 0 / 180 * pi])

    q = np.zeros((6,))

    q -= np.array(Theta0)

    # Define parameters
    max_iterations = 1000
    step_size = 0.01
    lambda_val = 0.1
    limit = 100
    rejcount = 0
    W = np.eye(6)  # 6x6 identity matrix
    W3 = np.eye(3)  # 3x3 identity matrix

    # Tolerance
    tolerance = 1e-6

    for i in range(1, max_iterations + 1):
        # Compute current forward kinematics
        T_current = forward_kinematics_Left(q, L, D, Phi)

        # Compute error
        TD = np.linalg.inv(T_current) @ T_desired
        S_block = TD[:3, :3]
        S = S_block - W3
        Vex = np.array([0.5 * (S[2, 1] - S[1, 2]),
                        0.5 * (S[0, 2] - S[2, 0]),
                        0.5 * (S[1, 0] - S[0, 1])])
        error = np.concatenate((TD[:3, 3], Vex))

        # Check convergence condition
        errorW = W @ error
        if np.linalg.norm(errorW) < tolerance:
            # print("Reaching the convergence condition")
            break

        # Compute Jacobian matrix
        J = compute_jacobian_Left(q, L, D, Phi)
        JtJ = J.T @ W @ J

        # Compute joint update
        delta_q = np.linalg.inv(JtJ + lambda_val * np.eye(JtJ.shape[0])) @ J.T @ W @ error
        q_new = q + delta_q.T

        # Compute new error
        TD_new = np.linalg.inv(forward_kinematics_Left(q_new, L, D, Phi)) @ T_desired
        S2 = TD_new[:3, :3] - np.eye(3)
        Vex2 = np.array([0.5 * (S2[2, 1] - S2[1, 2]),
                        0.5 * (S2[0, 2] - S2[2, 0]),
                        0.5 * (S2[1, 0] - S2[0, 1])])
        error_new = np.concatenate((TD_new[:3, 3], Vex2))

        errorNewW = W @ error_new

        if np.linalg.norm(errorNewW) < np.linalg.norm(errorW):
            q = q_new
            error = error_new
            lambda_val /= 2
            rejcount = 0
        else:
            lambda_val *= 2
            rejcount += 1
            if rejcount > limit:
                print("Unable to solve")
                break

            continue

        q += step_size * delta_q
    
    q += np.array(Theta0)

    return q