import numpy as np
import torch
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
torch.set_printoptions(sci_mode=False, precision=6, threshold=float('inf'))

def lagrangian_function(x, Y):
    return f(x) + Y @ (A @ x - b) + rho / 2 * ((A @ x - b)**2).sum()


def f(x):
    U, D, V, W, Ci, Ui, Qi = convert_x_to_matricies(x)
    resembled_x = resemble_matrix(U, D, V)
    resembled_y = resemble_matrix(U, D, W)
    tensor_x_loss = torch.norm(tensor_x - resembled_x, p=2)
    tensor_y_loss = torch.norm(tensor_y - resembled_y, p=2)
    side_information_loss = side_info_opt_func_val(Sa, Ci, Ui, Qi, U)
    return tensor_x_loss + tensor_y_loss + side_information_loss


def convert_x_to_matricies(x):
    num_si = len(Sa)
    base_length = num_drug * rank * 2 + num_disease * rank + num_ddi * rank
    si_length = rank * num_drug * 2
    U = x[:rank * num_drug].reshape(num_drug, rank)
    D = x[rank * num_drug:rank * num_drug * 2].reshape(num_drug, rank)
    V = x[rank * num_drug * 2: rank * num_drug * 2 + num_disease * rank].reshape(num_disease, rank)
    W = x[rank * num_drug * 2 + num_disease * rank: base_length].reshape(num_ddi, rank)
    Ci = []
    Ui = []
    Qi = []
    for i in range(num_si):
        Ci.append(x[base_length + i * si_length: base_length + i * si_length + rank * num_drug].reshape(num_drug, rank))
        Ui.append(x[base_length + i * si_length + rank * num_drug: base_length + i * si_length + si_length].reshape(num_drug, rank))
        Qi.append(calculate_Qi(Ui[i]))

    return U, D, V, W, Ci, Ui, Qi


def side_info_opt_func_val(Sa, Ci, Ui, Qi, U):

    result = 0.0
    for i in range(len(Sa)):
        temp = torch.norm(Sa[i] - Ci[i] @ Ui[i].t(), p=2) + torch.norm(Ci[i] @ Qi[i] - U, p=2)
        result += temp

    return result


def resemble_matrix(U, D, V):
    result = torch.zeros(U.shape[0], U.shape[0], V.shape[0]).to(device)
    num_col = U.shape[1]
    for i in range(num_col):
        Ui = U[:, i]
        Di = D[:, i]
        Vi = V[:, i]
        result = result + three_way_outer_product(Ui, Di, Vi)

    return result


def three_way_outer_product(a, b, c):
    return torch.einsum('i,j,k', a, b, c)


def generate_A_x_b():
    # calculate the row num and col num of the A matrix
    col_length = rank * num_drug * 2 + rank * num_disease + rank * num_ddi
    base_length = col_length
    for i in range(num_si):
        col_length += num_drug * rank * 2
    num_constraint = 1 + num_si

    # initialize A matrix
    A = torch.zeros(num_constraint, col_length, requires_grad=False, dtype=torch.float32).to(device)
    # assign value for the first constraint: U = D
    A[0, :rank * num_drug] = 1
    A[0, rank * num_drug : rank * num_drug * 2] = -1

    # si length is number of elements of one matrix in ci pluse one matrix in ui
    si_length = num_drug * rank * 2

    # assign value for rest of constraints: C[i] = U[i]
    for i in range(num_si):
        constraint_index = i + 1
        start_index = base_length + i * si_length
        A[constraint_index, start_index: (start_index + int(si_length / 2))] = 1
        A[constraint_index, (start_index + int(si_length / 2)): (start_index + si_length)] = -1

    # Concatenate all flattened tensors into a single 1D tensor
    x = torch.rand(col_length, dtype=torch.float32, requires_grad=False).to(device)

    b = torch.zeros(num_constraint, requires_grad=False, dtype=torch.float32).to(device)

    return A, x, b


def update_x(x, Y):
    """ update x with gradient descent """
    # decide the alternating two parts: matrix U,D,V,W, and C(i) matrices and U(i) matrices

    # decide the total number of components of U, D, V, W
    UDVW_length = num_drug * rank * 2 + num_disease * rank + num_ddi * rank

    # components that need to calculate partial derivative
    x_UDVW = x[:UDVW_length].clone().detach().requires_grad_(True)

    # temp x where we input it into lagragian function
    x_UDVW_temp = x.clone().detach()

    # change the value so this part can calculate gradient
    x_UDVW_temp[:UDVW_length] = x_UDVW

    # function value
    UDVW_lagrangian = lagrangian_function(x_UDVW_temp, Y)

    # calculate gradient
    grad_x_UDVW = torch.autograd.grad(UDVW_lagrangian, x_UDVW, create_graph=True)[0]

    # updated x values for UDVW
    x_UDVW_new = x_UDVW_temp[:UDVW_length] - lr * grad_x_UDVW

    # update x
    x.data[:UDVW_length] = torch.clamp(x_UDVW_new, min=0)

    # repeat the prvious steps, but for the x values of Ci and Ui
    x_CiUi = x[UDVW_length:].clone().detach().requires_grad_(True)
    x_CiUi_temp = x.clone().detach()
    x_CiUi_temp[UDVW_length:] = x_CiUi
    CiUi_lagrangian = lagrangian_function(x_CiUi_temp, Y)
    grad_x_CiUi = torch.autograd.grad(CiUi_lagrangian, x_CiUi, create_graph=True)[0]
    x_CiUi_new = x_CiUi_temp[UDVW_length:] - lr * grad_x_CiUi
    x.data[UDVW_length:] = torch.clamp(x_CiUi_new, min=0)


def update_lambda(Y):
    new_lambda = Y + rho * (A @ x - b)
    Y.data = new_lambda

def losses_test():
    U, D, V, W, _, _, _ = convert_x_to_matricies(x)
    real_tensor_val_x = real_tensor_x[tuple(x_test_indices.t())]
    resembled_matrix_val_x = resemble_matrix(U, D, V)[tuple(x_test_indices.t())]

    real_tensor_val_y = real_tensor_y[tuple(y_test_indices.t())]
    resembled_matrix_val_y = resemble_matrix(U, D, W)[tuple(y_test_indices.t())]

    return torch.norm(real_tensor_val_x - resembled_matrix_val_x, p=2).item(),\
           torch.norm(real_tensor_val_y - resembled_matrix_val_y, p=2).item()


def pprint(i, x, Y):
    loss = f(x)
    augmented_function_loss = lagrangian_function(x, Y)
    x_test_loss, y_test_loss = losses_test()

    print(
        f'\n{i+1}th iter, L:{augmented_function_loss:.2f}, f: {loss:.2f},'
        f' x_test_loss: {x_test_loss:.2f}, y_test_loss: {y_test_loss:.2f}'
    )
    # print(f'x: {x}')
    print(f'multiplier: {Y}')
    print("constraints violation: ")
    print(A @ x - b)

    return loss, augmented_function_loss, x_test_loss, y_test_loss


def solve(x, Y, iteration):
    losses_df = pd.DataFrame(columns=["iteration", "loss", "augmented_lagrangian_loss", "x_test_loss", "y_test_loss"])

    for i in range(iteration):
        update_x(x, Y)
        update_lambda(Y)

        # return and save the losses
        loss, alf_loss, x_test_loss, y_test_loss = pprint(i, x, Y)
        new_row = {"iteration": i, "loss": loss.item(), "augmented_lagrangian_loss": alf_loss.item(),
                   "x_test_loss": x_test_loss, "y_test_loss": y_test_loss}

        # save the result every 5 iterations (also saves the result the first iteration)
        losses_df = pd.concat([losses_df, pd.DataFrame([new_row])], ignore_index=True)
        if i % 5 == 0 or i == (iteration - 1):
            # paths for saving the result and loss
            x_path = full_save_dir + ".pt"
            loss_path = full_save_dir + ".csv"
            # save the result and loss
            torch.save(x, x_path)
            losses_df.to_csv(loss_path, index=False)

            print(f"Saved at iteration: {i} - Latest tensor X overwritten. loss saved.")


def load_tensor_x_y(base_dir):
    with open(base_dir + 'tensor_x.pickle', 'rb') as t_x:
        x = pickle.load(t_x)
    with open(base_dir + 'tensor_y.pickle', 'rb') as t_y:
        y = pickle.load(t_y)

    tensor_x = [x[key] for key in sorted(x.keys())]  # Extract arrays in key order
    # Step 3: Stack the arrays along the third axis (axis=2)
    tensor_x = torch.tensor(np.stack(tensor_x, axis=2), dtype=torch.float32)

    tensor_y = [y[key] for key in sorted(y.keys())]  # Extract arrays in key order
    # Step 3: Stack the arrays along the third axis (axis=2)
    tensor_y = torch.tensor(np.stack(tensor_y, axis=2), dtype=torch.float32)

    return tensor_x.to(device), tensor_y.to(device)


def load_si(base_dir):
    Sa = []
    for i in si:
        print(base_dir + 'si_{}.csv'.format(i))
        temp_sa = torch.tensor(pd.read_csv(base_dir + 'si_{}.csv'.format(i)).fillna(0).to_numpy()[:, 1:],
                               dtype=torch.float32).to(device)
        Sa.append(temp_sa)

    return Sa


def calculate_Qi(Ui):
    val = []
    result = torch.zeros((Ui.shape[1], Ui.shape[1]), dtype=torch.float32, requires_grad=False)
    for i in range(Ui.shape[1]):
        value_to_add = torch.sum(Ui[:, i])
        val.append(value_to_add)
    result.diagonal().copy_(torch.tensor(val, dtype=torch.float32))
    return result.to(device)


def generate_test_tensor(tensor, test_ratio, rnd_seed):
    # Get the indices of `1` and `0` values
    one_indices = (tensor == 1).nonzero(as_tuple=False)
    zero_indices = (tensor == 0).nonzero(as_tuple=False)

    # Determine the number of `1` and `0` values to replace
    num_ones_to_replace = int(test_ratio * one_indices.size(0))
    num_zeros_to_replace = num_ones_to_replace

    # Create a generator for reproducibility (optional)
    gen = torch.Generator()
    gen.manual_seed(rnd_seed)

    # Randomly select indices of `1` values
    selected_one_indices = one_indices[torch.randperm(one_indices.size(0), generator=gen)[:num_ones_to_replace]]
    # Randomly select indices of `0` values
    selected_zero_indices = zero_indices[torch.randperm(zero_indices.size(0), generator=gen)[:num_zeros_to_replace]]

    # Replace selected indices in the original tensor with `0.5`
    modified_tensor = tensor.clone()  # Create a copy to avoid modifying the original tensor

    modified_tensor[tuple(selected_one_indices.t())] = 0
    modified_tensor[tuple(selected_zero_indices.t())] = 0

    # Combine the selected indices for the output
    selected_indices = torch.cat((selected_one_indices, selected_zero_indices), dim=0)

    return modified_tensor, selected_indices


def result_to_csv(real_x, real_y, pred_x, pred_y, x_test_indices, y_test_indices):
    # Extract values at the indices of interest from the original and modified tensors
    real_x_values = real_x[tuple(x_test_indices.t())]
    real_y_values = real_y[tuple(y_test_indices.t())]

    pred_x_values = pred_x[tuple(x_test_indices.t())]
    pred_y_values = pred_y[tuple(y_test_indices.t())]

    # Create a dataframe to store the indices and values
    data_x = {
        "Indices": [tuple(idx.tolist()) for idx in x_test_indices],
        "label": real_x_values.tolist(),
        "prediction": pred_x_values.tolist()
    }
    df_x = pd.DataFrame(data_x)

    data_y = {
        "Indices": [tuple(idx.tolist()) for idx in y_test_indices],
        "label": real_y_values.tolist(),
        "prediction": pred_y_values.tolist()
    }
    df_y = pd.DataFrame(data_y)

    # Save the dataframe to a CSV file
    csv_file_path_x = "{}test_x_result.csv".format(full_save_dir)
    df_x.to_csv(csv_file_path_x, index=False)
    csv_file_path_y = "{}test_y_result.csv".format(full_save_dir)
    df_y.to_csv(csv_file_path_y, index=False)

    return df_x, df_y


if __name__ == '__main__':
    # check if gpu is used
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # weather to train or to visualize the result
    train = False

    # basic parameter of input data
    rank = 3
    num_drug = 225
    num_disease = 19
    num_ddi = 4
    si = [0, 1, 2]
    num_si = len(si)

    # the root directery to save the results
    base_dir = '../data/'

    # the directory and file name we are going to save the losses and result
    save_name = ''
    for index in si:
        save_name += str(index) + '_'
    save_file_ending = '_{}si'.format(save_name)

    save_dir = '../output/'
    full_save_dir = save_dir + save_name

    # load up the tensors
    real_tensor_x, real_tensor_y = load_tensor_x_y(base_dir)

    # generate the test tensor and save the indicies
    tensor_x, x_test_indices = generate_test_tensor(tensor=real_tensor_x, test_ratio=0.1, rnd_seed=123)
    tensor_y, y_test_indices = generate_test_tensor(tensor=real_tensor_y, test_ratio=0.1, rnd_seed=123)

    # load up the side information
    Sa = load_si(base_dir)
    # learning rate
    lr = torch.tensor(0.0008, dtype=torch.float32).to(device)
    # penalty parameter
    rho = torch.tensor(1, dtype=torch.float32).to(device)

    if train:
        # initial value of lagragian multiplier
        Y = torch.zeros(1 + num_si, dtype=torch.float32).to(device)

        # initialize the A, x and b
        A, x, b = generate_A_x_b()

        solve(x, Y, iteration=1300)
    else:
        x = torch.load(full_save_dir + '.pt')
        U, D, V, W, Ci, Ui, Qi = convert_x_to_matricies(x)
        pred_x = resemble_matrix(U, D, V)
        pred_y = resemble_matrix(U, D, W)
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