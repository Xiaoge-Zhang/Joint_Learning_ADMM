import numpy as np
import pandas as pd
import pickle
# this is just s script that output the stats of the dataset

def sparsity(tensor):
    # Calculate the total number of elements
    total_elements = tensor.size

    # Count the number of zeros
    one_elements = np.count_nonzero(tensor == 1)

    # Calculate sparsity
    return one_elements / total_elements, one_elements


def side_info_coverage(si):
    count=0
    for i in range(si.shape[1]):
        # Get the column and check if it matches the pattern: 1 on diagonal, 0 elsewhere
        if np.count_nonzero(si[:, i]) == 1 and si[i, i] == 1:
            count += 1

    return si.shape[1], si.shape[1] - count

base_dir = '../data/'

with open(base_dir + 'tensor_x.pickle', 'rb') as t_x:
    x = pickle.load(t_x)
with open(base_dir + 'tensor_y.pickle', 'rb') as t_y:
    y = pickle.load(t_y)

tensor_x = [x[key] for key in sorted(x.keys())]  # Extract arrays in key order
tensor_x = np.stack(tensor_x, axis=2, dtype=float)

tensor_y = [y[key] for key in sorted(y.keys())]  # Extract arrays in key order
tensor_y = np.stack(tensor_y, axis=2, dtype=float)

print("Sparsity of X: {}, Sparsity of Y:{}".format(sparsity(tensor_x), sparsity(tensor_y)))

si = [0, 1, 2, 3, 4]
for i in si:
    print(base_dir + 'si_{}.csv'.format(i))
    temp_sa = pd.read_csv(base_dir + 'si_{}.csv'.format(i)).fillna(0).to_numpy()[:, 1:]
    total_col, si_col = side_info_coverage(temp_sa)
    print("side information {}, {} durgs have side information out of {} drugs".format(i, si_col, total_col))

