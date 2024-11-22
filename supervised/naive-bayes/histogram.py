
import numpy as np

matrix = np.array([
    [1, 2, 2, 3],
    [1, 2, 2, 3],
    [6, 7, 7, 6],
    [1, 2, 2, 3],
    [1, 9, 2, 3],
    [1, 2, 2, 3],
    [1, 2, 2, 3],
    [1, 2, 2, 3],
])

def calculate_probs(arr, proportion_limit, kernel_func=''):
    probs = []
    for col in arr.T:    
        if is_discrete(col, proportion_limit):
            p, l= calculate_discrete_probs(col)
            probs.append(dict(map(reversed, zip(p,l))))
        else:
            print('naure')
            pass
    return probs

def calculate_discrete_probs(arr):
    labels, counts = np.unique(arr, return_counts=True)
    return (counts / len(arr), labels) 

def is_discrete(arr, proportion_limit):
    value_count = len(arr)
    unique_count = len(np.unique(arr))
    return (unique_count / value_count) <= proportion_limit


histograms = calculate_probs(matrix, 0.4)
print(histograms)
# Print the results
for i, hist in enumerate(histograms):
    print(f"Row {i} histogram: {hist}")
