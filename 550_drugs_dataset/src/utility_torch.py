import matplotlib.pyplot as plt
import torch
from sklearn import metrics
import pandas as pd
import numpy as np
import pickle


class Utility_torch:
    def __init__(self, tensor_x, tensor_y, Sa,
                 A, b, si_weight,
                 lr, rho, rank, num_drug, num_disease, num_ddi, num_si,
                 device, base_dir, full_save_dir, rnd_seed):
        self.tensor_x = tensor_x
        self.tensor_y = tensor_y
        self.Sa = Sa
        self.A = A
        self.b = b
        self.si_weight = si_weight
        self.lr = lr
        self.rho = rho
        self.rank = rank
        self.num_drug = num_drug
        self.num_disease = num_disease
        self.num_ddi = num_ddi
        self.num_si = num_si
        self.device = device
        self.base_dir = base_dir
        self.full_save_dir = full_save_dir
        self.rnd_seed = rnd_seed

    def lagrangian_function(self, x, Y):
        return f(x) + Y @ (self.A @ x - self.b) + self.rho / 2 * ((self.A @ x - self.b) ** 2).sum()

    def f(self, x):
        U, D, V, W, Ci, Ui, Qi = self.convert_x_to_matricies(x)
        resembled_x = self.resemble_matrix(U, D, V)
        resembled_y = self.resemble_matrix(U, D, W)
        tensor_x_loss = torch.norm(self.tensor_x - resembled_x, p=2)
        tensor_y_loss = torch.norm(self.tensor_y - resembled_y, p=2)
        side_information_loss = self.side_info_opt_func_val(Ci, Ui, Qi, U)
        return tensor_x_loss + tensor_y_loss + side_information_loss

    def update_partial_x(self, x, Y, start_index, end_index):
        # components that need to calculate partial derivative
        x_partial = x[start_index:end_index].clone().detach().requires_grad_(True)

        # temp x where we input it into lagragian function
        x_temp = x.clone().detach()

        # change the value so this part can calculate gradient
        x_temp[start_index:end_index] = x_partial

        # function value
        lagrangian_val = self.lagrangian_function(x_temp, Y)

        # calculate gradient
        grad_x = torch.autograd.grad(lagrangian_val, x_partial, create_graph=True)[0]

        # updated x values for UDVW
        x_temp_new = x_temp[start_index:end_index] - self.lr * grad_x

        # update x
        x.data[start_index:end_index] = torch.clamp(x_temp_new, min=0)

    def side_info_opt_func_val(self, Ci, Ui, Qi, U):
        result = 0.0
        for i in range(self.num_si):
            temp = self.si_weight[i] * (torch.norm(self.Sa[i] - Ci[i] @ Ui[i].t(), p=2) + torch.norm(Ci[i] @ Qi[i] - U, p=2))
            result += temp

        return result

    def convert_x_to_matricies(self, x):
        base_length = self.num_drug * self.rank * 2 + self.num_disease * self.rank + self.num_ddi * self.rank
        si_length = self.rank * self.num_drug * 2
        U = x[:self.rank * self.num_drug].reshape(self.num_drug, self.rank)
        D = x[self.rank * self.num_drug:self.rank * self.num_drug * 2].reshape(self.num_drug, self.rank)
        V = x[self.rank * self.num_drug * 2: self.rank * self.num_drug * 2 + self.num_disease * self.rank].reshape(self.num_disease, self.rank)
        W = x[self.rank * self.num_drug * 2 + self.num_disease * self.rank: base_length].reshape(self.num_ddi, self.rank)
        Ci = []
        Ui = []
        Qi = []
        for i in range(self.num_si):
            Ci.append(
                x[base_length + i * si_length: base_length + i * si_length + self.rank * self.num_drug].reshape(self.num_drug, self.rank))
            Ui.append(x[base_length + i * si_length + self.rank * self.num_drug: base_length + i * si_length + si_length].reshape(
                self.num_drug, self.rank))
            Qi.append(self.calculate_Qi(Ui[i]))

        return U, D, V, W, Ci, Ui, Qi

    def generate_A_x_b(self):
        # calculate the row num and col num of the A matrix
        col_length = self.rank * self.num_drug * 2 + self.rank * self.num_disease + self.rank * self.num_ddi
        base_length = col_length
        for i in range(self.num_si):
            col_length += self.num_drug * self.rank * 2
        num_constraint = 1 + self.num_si

        # initialize A matrix
        A = torch.zeros(num_constraint * self.rank * self.num_drug, col_length, requires_grad=False, dtype=torch.float32).to(
            self.device)
        # assign value for the first constraint: U = D
        U_start_index = 0
        D_start_index = self.rank * self.num_drug
        for i in range(self.rank * self.num_drug):
            A[i, U_start_index + i] = 1
            A[i, D_start_index + i] = -1

        # si length is number of elements of one matrix in ci pluse one matrix in ui
        si_length = self.num_drug * self.rank * 2

        # assign value for rest of constraints: C[i] = U[i]
        for i in range(self.num_si):
            row_starting_index = (1 + i) * self.num_drug * self.rank
            ci_starting_index = base_length + i * 2 * self.num_drug * self.rank
            ui_starting_index = ci_starting_index + self.num_drug * self.rank
            for i in range(self.rank * self.num_drug):
                A[row_starting_index + i, ci_starting_index + i] = 1
                A[row_starting_index + i, ui_starting_index + i] = -1

        # Concatenate all flattened tensors into a single 1D tensor
        x = torch.rand(col_length, dtype=torch.float32, requires_grad=False).to(self.device)

        b = torch.zeros(num_constraint * self.rank * self.num_drug, requires_grad=False, dtype=torch.float32).to(self.device)

        return A, x, b

    def resemble_matrix(self, U, D, V):
        result = torch.zeros(U.shape[0], U.shape[0], V.shape[0]).to(self.device)
        num_col = U.shape[1]
        for i in range(num_col):
            Ui = U[:, i]
            Di = D[:, i]
            Vi = V[:, i]
            result = result + self.three_way_outer_product(Ui, Di, Vi)

        return result

    def three_way_outer_product(self, a, b, c):
        return torch.einsum('i,j,k', a, b, c)

    def calculate_Qi(self, Ui):
        val = []
        result = torch.zeros((Ui.shape[1], Ui.shape[1]), dtype=torch.float32, requires_grad=False)
        for i in range(Ui.shape[1]):
            value_to_add = torch.sum(Ui[:, i])
            val.append(value_to_add)
        result.diagonal().copy_(torch.tensor(val, dtype=torch.float32))
        return result.to(self.device)


class File_IO_torch:
    def __init__(self, device, base_dir, full_save_dir, rnd_seed):
        self.device = device
        self.base_dir = base_dir
        self.full_save_dir = full_save_dir
        self.rnd_seed = rnd_seed

    def load_si(self, si):
        Sa = []
        for i in si:
            print(self.base_dir + 'si_{}.csv'.format(i))
            temp_sa = torch.tensor(pd.read_csv(self.base_dir + 'si_{}.csv'.format(i)).fillna(0).to_numpy()[:, 1:],
                                   dtype=torch.float32).to(self.device)
            Sa.append(temp_sa)

        return Sa

    def load_tensor_x_y(self):
        with open(self.base_dir + 'tensor_x.pickle', 'rb') as t_x:
            x = pickle.load(t_x)
        with open(self.base_dir + 'tensor_y.pickle', 'rb') as t_y:
            y = pickle.load(t_y)

        tensor_x = [x[key] for key in sorted(x.keys())]  # Extract arrays in key order
        # Step 3: Stack the arrays along the third axis (axis=2)
        tensor_x = torch.tensor(np.stack(tensor_x, axis=2), dtype=torch.float32)

        tensor_y = [y[key] for key in sorted(y.keys())]  # Extract arrays in key order
        # Step 3: Stack the arrays along the third axis (axis=2)
        tensor_y = torch.tensor(np.stack(tensor_y, axis=2), dtype=torch.float32)

        return tensor_x.to(self.device), tensor_y.to(self.device)

    def result_to_csv(self, real_x, real_y, pred_x, pred_y, x_test_indices, y_test_indices):
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
        csv_file_path_x = "{}test_x_result_{}.csv".format(self.full_save_dir, self.rnd_seed)
        df_x.to_csv(csv_file_path_x, index=False)
        csv_file_path_y = "{}test_y_result_{}.csv".format(self.full_save_dir, self.rnd_seed)
        df_y.to_csv(csv_file_path_y, index=False)

        return df_x, df_y


def plot_roc_aupr(x_real, x_pred, y_real, y_pred):
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


def generate_test_tensor(tensor, test_ratio, rnd_seed, missing_rate=0.0):
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
    randomized_one_indicies = one_indices[torch.randperm(one_indices.size(0), generator=gen)]
    selected_one_indices = randomized_one_indicies[:num_ones_to_replace]

    # Randomly select indices of `0` values
    selected_zero_indices = zero_indices[torch.randperm(zero_indices.size(0), generator=gen)[:num_zeros_to_replace]]

    # Replace selected indices in the original tensor with `0.5`
    modified_tensor = tensor.clone()  # Create a copy to avoid modifying the original tensor

    modified_tensor[tuple(selected_one_indices.t())] = 0
    modified_tensor[tuple(selected_zero_indices.t())] = 0

    # select values to mask if missing rate is bigger than 0
    if missing_rate > 0.0:
        num_ones_to_mask = int(missing_rate * (one_indices.size(0) - num_ones_to_replace * 2))
        masked_one_indicies = randomized_one_indicies[num_ones_to_replace:num_ones_to_replace + num_ones_to_mask]
        modified_tensor[tuple(masked_one_indicies.t())] = 0

    # Combine the selected indices for the output
    selected_indices = torch.cat((selected_one_indices, selected_zero_indices), dim=0)

    return modified_tensor, selected_indices
