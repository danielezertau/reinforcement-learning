import numpy as np


def int_to_char(n):
    chars = ['B', 'K', 'O', '-']
    return chars[n]


if __name__ == '__main__':
    p = [[0.1, 0.325, 0.25, 0.325],  # B
         [0.4, 0, 0.4, 0.2],  # K
         [0.2, 0.2, 0.2, 0.4],  # O
         [1, 0, 0, 0]]  # -
    p = np.array(p)
    C = np.zeros((5, 3))
    # init
    C[-1] = p[:3, -1].T

    # Build value matrix
    for i in range(3, -1, -1):
        for l in range(3):
            C[i][l] = np.max(p[l, :3] * C[i + 1, :3])

    # Backtrack to find the solution
    for i in range(5):
        print(int_to_char(np.argmax(C[i, :])), end='')
