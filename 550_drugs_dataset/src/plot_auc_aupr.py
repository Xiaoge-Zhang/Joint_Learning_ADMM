import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve


# Function to calculate AUC and AUPR for a set of test files
def calculate_metrics(test_files):
    fpr_list, tpr_list, precision_list, recall_list = [], [], [], []
    aucs, auprs = [], []

    for file in test_files:
        df = pd.read_csv(file)
        fpr, tpr, _ = roc_curve(df['label'], df['prediction'])
        precision, recall, _ = precision_recall_curve(df['label'], df['prediction'])

        # Plot Precision-Recall curve for x and y
        # plt.figure()
        # plt.plot(recall, precision, label="x tensor, AUPR", c='red')
        # plt.legend(loc=0)
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.title('Precision-Recall Curve')
        # plt.show()
        # print(recall, precision)
        # jjj
        #
        # print(precision)
        # print(recall)
        # Calculate AUC and AUPR
        roc_auc = auc(fpr, tpr)
        pr_auc = auc(recall, precision)
        # print('roc_auc', roc_auc)
        # print('pr_auc', pr_auc)

        fpr_list.append(fpr)
        tpr_list.append(tpr)
        precision_list.append(precision)
        recall_list.append(recall)

        aucs.append(roc_auc)
        auprs.append(pr_auc)

    return np.array(aucs), np.array(auprs), fpr_list, tpr_list, precision_list, recall_list

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def plot_combined_metrics_with_std(aucs_x, auprs_x, fpr_list_x, tpr_list_x, precision_list_x, recall_list_x,
                                   aucs_y, auprs_y, fpr_list_y, tpr_list_y, precision_list_y, recall_list_y):
    # Set font size for the whole plot
    font_size = 28
    plt.rcParams.update({'font.size': font_size})

    plt.figure(figsize=(20, 8))

    # Plot ROC curves for both x and y datasets
    plt.subplot(1, 2, 1)
    mean_fpr = np.linspace(0, 1, 100)

    # Interpolate and calculate mean/std for x
    tpr_interp_x = [np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(fpr_list_x, tpr_list_x)]
    mean_tpr_x = np.mean(tpr_interp_x, axis=0)
    std_tpr_x = np.std(tpr_interp_x, axis=0)

    # Interpolate and calculate mean/std for y
    tpr_interp_y = [np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(fpr_list_y, tpr_list_y)]
    mean_tpr_y = np.mean(tpr_interp_y, axis=0)
    std_tpr_y = np.std(tpr_interp_y, axis=0)

    # Plot ROC for x
    plt.plot(mean_fpr, mean_tpr_x, label=f'Mean ROC X (AUC = {np.mean(aucs_x):.4f})', color='blue')
    plt.fill_between(mean_fpr, mean_tpr_x - std_tpr_x, mean_tpr_x + std_tpr_x, alpha=0.2, color='blue')

    # Plot ROC for y
    plt.plot(mean_fpr, mean_tpr_y, label=f'Mean ROC Y (AUC = {np.mean(aucs_y):.4f})', color='orange')
    plt.fill_between(mean_fpr, mean_tpr_y - std_tpr_y, mean_tpr_y + std_tpr_y, alpha=0.2, color='orange')
    plt.grid()
    plt.title('ROC Curves with Std', fontsize=font_size)
    plt.xlabel('False Positive Rate', fontsize=font_size)
    plt.ylabel('True Positive Rate', fontsize=font_size)
    plt.legend(fontsize=24)

    # Plot Precision-Recall curves for both x and y datasets
    plt.subplot(1, 2, 2)

    # Define a common recall grid for interpolation
    mean_recall = np.linspace(0, 1, 100)

    # Interpolate precision values onto the common recall grid for X dataset
    precision_interp_x = [np.interp(mean_recall, np.flip(recall), np.flip(precision)) for recall, precision in zip(recall_list_x, precision_list_x)]
    mean_precision_x = np.mean(precision_interp_x, axis=0)
    std_precision_x = np.std(precision_interp_x, axis=0)

    # Plot Precision-Recall for X dataset with mean and std
    plt.plot(mean_recall, mean_precision_x, label=f'Mean Precision-Recall X (AUPR = {np.mean(auprs_x):.4f})', color='blue')
    plt.fill_between(mean_recall, mean_precision_x - std_precision_x, mean_precision_x + std_precision_x, color='blue', alpha=0.2)

    # Interpolate precision values onto the common recall grid for Y dataset
    precision_interp_y = [np.interp(mean_recall, np.flip(recall), np.flip(precision)) for recall, precision in zip(recall_list_y, precision_list_y)]
    mean_precision_y = np.mean(precision_interp_y, axis=0)
    std_precision_y = np.std(precision_interp_y, axis=0)

    # Plot Precision-Recall for Y dataset with mean and std
    plt.plot(mean_recall, mean_precision_y, label=f'Mean Precision-Recall Y (AUPR = {np.mean(auprs_y):.4f})', color='orange')
    plt.fill_between(mean_recall, mean_precision_y - std_precision_y, mean_precision_y + std_precision_y, color='orange', alpha=0.2)

    plt.title('Precision-Recall Curves with Std', fontsize=font_size)
    plt.xlabel('Recall', fontsize=font_size)
    plt.ylabel('Precision', fontsize=font_size)
    plt.legend(fontsize=24)
    plt.grid()

    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig('auc_aupr.pdf', format='pdf')
    plt.show()



# List of x and y test result files
test_files_x = ['results/0_1_2_3_4_test_x_result_123.csv', 'results/0_1_2_3_4_test_x_result_124.csv', 'results/0_1_2_3_4_test_x_result_125.csv',
                'results/0_1_2_3_4_test_x_result_126.csv', 'results/0_1_2_3_4_test_x_result_127.csv']
test_files_y = ['results/0_1_2_3_4_test_y_result_123.csv', 'results/0_1_2_3_4_test_y_result_124.csv', 'results/0_1_2_3_4_test_y_result_125.csv',
                'results/0_1_2_3_4_test_y_result_126.csv', 'results/0_1_2_3_4_test_y_result_127.csv']

# Calculate metrics for x and y test sets
aucs_x, auprs_x, fpr_list_x, tpr_list_x, precision_list_x, recall_list_x = calculate_metrics(test_files_x)
aucs_y, auprs_y, fpr_list_y, tpr_list_y, precision_list_y, recall_list_y = calculate_metrics(test_files_y)

# Plot combined ROC and Precision-Recall curves for x and y datasets
plot_combined_metrics_with_std(aucs_x, auprs_x, fpr_list_x, tpr_list_x, precision_list_x, recall_list_x,
                               aucs_y, auprs_y, fpr_list_y, tpr_list_y, precision_list_y, recall_list_y)