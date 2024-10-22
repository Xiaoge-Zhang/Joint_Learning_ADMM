import random

import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker
import matplotlib.pyplot as plt
import pickle
import os
from sklearn import metrics
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def resemble_matrix(U, D, V):
    result = np.zeros((U.shape[0], U.shape[0], V.shape[0]), dtype=float)
    num_col = U.shape[1]
    for i in range(num_col):
        Ui = U[:, i]
        Di = D[:, i]
        Vi = V[:, i]
        result = result + three_way_outer_product(Ui, Di, Vi)

    return result


def three_way_outer_product(a, b, c):
    return np.einsum('i,j,k', a, b, c)

def load_tensor_x_y(base_dir):
    with open(base_dir + 'tensor_x.pickle', 'rb') as t_x:
        x = pickle.load(t_x)
    with open(base_dir + 'tensor_y.pickle', 'rb') as t_y:
        y = pickle.load(t_y)

    tensor_x = [x[key] for key in sorted(x.keys())]  # Extract arrays in key order
    # Step 3: Stack the arrays along the third axis (axis=2)
    tensor_x = np.stack(tensor_x, axis=2, dtype=float)

    tensor_y = [y[key] for key in sorted(y.keys())]  # Extract arrays in key order
    # Step 3: Stack the arrays along the third axis (axis=2)
    tensor_y = np.stack(tensor_y, axis=2, dtype=float)

    return tensor_x, tensor_y


def generate_test_tensor(tensor, test_ratio, rnd_seed, missing_rate=0):
    # Get the indices of `1` and `0` values
    one_indices = list(zip(*np.where(tensor == 1)))
    zero_indices = list(zip(*np.where(tensor == 0)))

    # Determine the number of `1` and `0` values to replace
    num_ones_to_replace = int(test_ratio * len(one_indices))
    num_zeros_to_replace = num_ones_to_replace

    # Randomly select indices of `1` values
    shuffled_one_indices = np.random.default_rng(seed=rnd_seed).permutation(len(one_indices))
    selected_one_indices = [one_indices[i] for i in shuffled_one_indices[:num_ones_to_replace]]
    selected_one_indices = tuple(zip(*selected_one_indices))


    # Randomly select indices of `0` values
    shuffled_zero_indices = np.random.permutation(len(zero_indices))
    selected_zero_indices = [zero_indices[i] for i in shuffled_zero_indices[:num_zeros_to_replace]]
    selected_zero_indices = tuple(zip(*selected_zero_indices))

    # Replace selected indices in the original tensor with `0.5`
    modified_tensor = tensor.copy()  # Create a copy to avoid modifying the original tensor

    modified_tensor[selected_one_indices] = 0
    modified_tensor[selected_zero_indices] = 0

    # Combine the selected indices for the output
    selected_indices = tuple(np.concatenate((idx1, idx0)) for idx1, idx0 in zip(selected_one_indices, selected_zero_indices))

    if missing_rate > 0.0:
        test_selected_indicies = list(zip(*selected_indices))

        # get all possible index
        indices = np.indices(tensor.shape)
        all_indices = [tuple(idx) for idx in np.vstack(map(np.ravel, indices)).T]

        # cut out the test entries
        possible_indices = list(set(all_indices)-set(test_selected_indicies))

        # select missing rate of possible entries
        random.seed(rnd_seed)
        cutted_indices = random.sample(possible_indices, int(missing_rate * len(possible_indices)))

        #turn them into 0
        modified_tensor[tuple(zip(*cutted_indices))] = 0

    return modified_tensor, selected_indices


def result_to_csv(real_x, real_y, pred_x, pred_y, x_test_indices, y_test_indices):
    # Extract values at the indices of interest from the original and modified tensors
    real_x_values = real_x[x_test_indices]
    real_y_values = real_y[y_test_indices]

    pred_x_values = pred_x[x_test_indices]
    pred_y_values = pred_y[y_test_indices]

    # Create a dataframe to store the indices and values
    data_x = {
        "Indices": [idx for idx in list(zip(*x_test_indices))],
        "label": real_x_values.tolist(),
        "prediction": pred_x_values.tolist()
    }
    df_x = pd.DataFrame(data_x)

    data_y = {
        "Indices": [idx for idx in list(zip(*y_test_indices))],
        "label": real_y_values.tolist(),
        "prediction": pred_y_values.tolist()
    }
    df_y = pd.DataFrame(data_y)

    return df_x, df_y


if __name__ == '__main__':
    base_dir = '../data/'

    np.set_printoptions(precision=2)
    # load up the tensors
    real_tensor_x, real_tensor_y = load_tensor_x_y(base_dir)

    # generate the test tensor and save the indicies
    tensor_x, x_test_indices = generate_test_tensor(tensor=real_tensor_x, test_ratio=0.1, rnd_seed=123, missing_rate=0.1)
    tensor_y, y_test_indices = generate_test_tensor(tensor=real_tensor_y, test_ratio=0.1, rnd_seed=123, missing_rate=0.1)

    # tensor generation
    tensor_x = tl.tensor(tensor_x)
    tensor_y = tl.tensor(tensor_y)
    rank = 3

    _, x_factors = tucker(tensor=tensor_x, rank=rank)
    _, y_factors = tucker(tensor=tensor_y, rank=rank)

    pred_x = resemble_matrix(x_factors[0], x_factors[1], x_factors[2])
    pred_y = resemble_matrix(y_factors[0], y_factors[1], y_factors[2])

    # result of the testing columns
    x_result, y_result = result_to_csv(real_tensor_x, real_tensor_y, pred_x, pred_y, x_test_indices, y_test_indices)

    # Ground truth of the testing cells
    x_real = np.array(x_result['label'].tolist())
    y_real = np.array(y_result['label'].tolist())

    # Prediction values of the testing cells
    x_pred = np.array(x_result['prediction'].tolist())
    y_pred = np.array(y_result['prediction'].tolist())

    # Return the values for graphing ROC curve
    fpr_x, tpr_x, thresholds_x = metrics.roc_curve(x_real, x_pred)
    fpr_y, tpr_y, thresholds_y = metrics.roc_curve(y_real, y_pred)

    # Generate and plot the ROC curve
    roc_auc1 = metrics.auc(fpr_x, tpr_x)
    roc_auc2 = metrics.auc(fpr_y, tpr_y)

    # Print AUC values
    print(f"AUC for x tensor: {roc_auc1}")
    print(f"AUC for y tensor: {roc_auc2}")

    # Plot ROC curve
    plt.plot(fpr_x, tpr_x, label="x tensor, AUC=" + str(roc_auc1), c='red')
    plt.plot(fpr_y, tpr_y, label="y tensor, AUC=" + str(roc_auc2), c='green')
    plt.legend(loc=0)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    # plt.savefig('roc_auc.png')
    plt.show()

    # Compute precision-recall and AUPR
    precision_x, recall_x, thresholds_pr_x = metrics.precision_recall_curve(x_real, x_pred)
    precision_y, recall_y, thresholds_pr_y = metrics.precision_recall_curve(y_real, y_pred)

    # Calculate AUPR for both x and y
    aupr_x = metrics.auc(recall_x, precision_x)
    aupr_y = metrics.auc(recall_y, precision_y)

    # Print AUPR values
    print(f"AUPR for x tensor: {aupr_x}")
    print(f"AUPR for y tensor: {aupr_y}")

    # Plot Precision-Recall curve for x and y
    plt.figure()
    plt.plot(recall_x, precision_x, label="x tensor, AUPR=" + str(aupr_x), c='red')
    plt.plot(recall_y, precision_y, label="y tensor, AUPR=" + str(aupr_y), c='green')
    plt.legend(loc=0)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    # plt.savefig('precision_recall_curve.png')
    plt.show()