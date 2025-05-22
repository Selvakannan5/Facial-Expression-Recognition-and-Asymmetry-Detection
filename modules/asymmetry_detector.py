import numpy as np

def check_asymmetry(landmarks):
    n = len(landmarks)
    left = landmarks[:n//2]
    right = landmarks[n//2:]
    left_mean = np.mean(left, axis=0)
    right_mean = np.mean(right, axis=0)
    diff = np.linalg.norm(left_mean - right_mean)
    return diff > 0.08