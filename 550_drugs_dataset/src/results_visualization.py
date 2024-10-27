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

        roc_auc = auc(fpr, tpr)
        pr_auc = auc(recall, precision)

        fpr_list.append(fpr)
        tpr_list.append(tpr)
        precision_list.append(precision)
        recall_list.append(recall)

        aucs.append(roc_auc)
        auprs.append(pr_auc)

    return np.array(aucs), np.array(auprs), fpr_list, tpr_list, precision_list, recall_list


def plot_combined_metrics_with_std(aucs, auprs, fpr_list, tpr_list, precision_list, recall_list,
                                   color, model_name, tensor_name):
    # Plot ROC curves for both x and y datasets
    plt.subplot(1, 2, 1)
    mean_fpr = np.linspace(0, 1, 100)

    # Interpolate and calculate mean/std for x
    tpr_interp = [np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(fpr_list, tpr_list)]
    mean_tpr = np.mean(tpr_interp, axis=0)
    std_tpr = np.std(tpr_interp, axis=0)

    # Plot ROC
    plt.plot(mean_fpr, mean_tpr, label='AUC = {} for {}'.format(np.round(np.mean(aucs), 4), model_name), color= color)
    # plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, alpha=0.2, color= color)

    plt.grid()
    plt.title('ROC Curves with Std of {}'.format(tensor_name), fontsize=font_size)
    plt.xlabel('False Positive Rate', fontsize=font_size)
    plt.ylabel('True Positive Rate', fontsize=font_size)
    plt.legend(fontsize=12)

    # Plot Precision-Recall curves for both x and y datasets
    plt.subplot(1, 2, 2)

    # Define a common recall grid for interpolation
    mean_recall = np.linspace(0, 1, 100)

    # Interpolate precision values onto the common recall grid
    precision_interp = [np.interp(mean_recall, np.flip(recall), np.flip(precision)) for recall, precision in zip(recall_list, precision_list)]
    mean_precision = np.mean(precision_interp, axis=0)
    std_precision = np.std(precision_interp, axis=0)


    # Plot Precision-Recall
    plt.plot(mean_recall, mean_precision, label='AUPR = {} for {}'.format(np.round(np.mean(auprs), 4), model_name), color=color)
    # plt.fill_between(mean_recall, mean_precision - std_precision, mean_precision + std_precision, color=color, alpha=0.2)

    plt.title('Precision-Recall Curves with Std of {}'.format(tensor_name), fontsize=font_size)
    plt.xlabel('Recall', fontsize=font_size)
    plt.ylabel('Precision', fontsize=font_size)
    plt.legend(fontsize=12)
    plt.grid()
    print(tensor_name)
    print(model_name)
    print("mean aucs: {}, std:{}".format(np.mean(aucs), np.std(aucs)))
    print("mean aupr: {}, std:{}".format(np.mean(auprs), np.std(auprs)))

save_dir = '../output/'

model_names = ['0_1_2_3_4', 'non_negative_tucker', 'constrained_parafac', 'tucker']

rnd_seeds = ['123', '124', '125', '126', '127']

colors = ['red', 'orange', 'blue', 'green']

# Set font size for the whole plot
font_size = 28
plt.rcParams.update({'font.size': font_size})

# plot metrics for X tensor
plt.figure(figsize=(20, 8))
tensor_name = "X"
for i, model_name in enumerate(model_names):
    test_files_x = []
    for rnd_seed in rnd_seeds:
        test_files_x.append(save_dir + model_name + "_test_x_result_" + rnd_seed + ".csv")

    aucs_x, auprs_x, fpr_list_x, tpr_list_x, precision_list_x, recall_list_x = calculate_metrics(test_files_x)
    # Plot combined ROC and Precision-Recall curves for x datasets
    if model_name == '0_1_2_3_4':
        model_name = "SI-ADMM"
    plot_combined_metrics_with_std(aucs_x, auprs_x, fpr_list_x, tpr_list_x, precision_list_x, recall_list_x, colors[i], model_name, tensor_name)

# Adjust layout and save figure
plt.tight_layout()
plt.savefig('auc_aupr_{}.pdf'.format(tensor_name), format='pdf')
plt.show()


# plot metrics on Y tensor
plt.figure(figsize=(20, 8))
tensor_name = "Y"
for i, model_name in enumerate(model_names):
    test_files_y = []
    for rnd_seed in rnd_seeds:
        test_files_y.append(save_dir + model_name + "_test_y_result_" + rnd_seed + ".csv")

    aucs_y, auprs_y, fpr_list_y, tpr_list_y, precision_list_y, recall_list_y = calculate_metrics(test_files_y)

    # Plot combined ROC and Precision-Recall curves for x datasets
    if model_name == '0_1_2_3_4':
        model_name = "SI-ADMM"
    plot_combined_metrics_with_std(aucs_y, auprs_y, fpr_list_y, tpr_list_y, precision_list_y, recall_list_y, colors[i], model_name, tensor_name)
# Adjust layout and save figure
plt.tight_layout()
plt.savefig('auc_aupr_{}.pdf'.format(tensor_name), format='pdf')
plt.show()
