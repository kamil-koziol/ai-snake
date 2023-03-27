from __future__ import annotations
import numpy as np

def relu(X: np.ndarray) -> np.ndarray:
    # relu => relu(x) = max(0, x)
    return np.maximum(0, X)

def softmax(X: np.ndarray) -> np.ndarray:
    # rozklad prawdopodobienstwa
    # !! - nie zrobilem implementacji safe numerical
    exp = X - X.max()
    sum_of_exp = np.sum(exp)
    return exp / sum_of_exp

def linear(X: np.ndarray) -> np.ndarray:
    # linear activation doesnt change antyhing
    return X

if __name__ == "__main__":
    # relu test

    x = np.array([-1,-2,-3,-4, 1,2,3,4])
    print(relu(x))
    print(softmax(x))
    print(linear(x))