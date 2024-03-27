import sys, os
from copy import deepcopy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('bmh')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
sys.path.append(os.environ.get("PROJECT_ROOT"))


def plot_calibration_curves(metrics_df, n_folds, output_folder_name):
    for i in range(n_folds):
        y_test = metrics_df[(metrics_df['fold_num'] == i) & (metrics_df['metric'] == 'y_test')]['val'].values[0]
        y_mean_test = np.round(
            metrics_df[(metrics_df['fold_num'] == i) & (metrics_df['metric'] == 'y_mean_test')]['val'].values[0],
            decimals=2)

        calibration_df = pd.DataFrame.from_dict({
            'y_test': y_test,
            'y_mean_test': y_mean_test,
        })

        decs = np.sort(calibration_df['y_mean_test'].unique())
        empirs = []
        counts = []
        for dec in decs:
            calibration_df[calibration_df['y_mean_test'] == dec].mean()
            empir = calibration_df[calibration_df['y_mean_test'] == dec].mean()['y_test']
            empirs.append(empir)
            counts.append(calibration_df[calibration_df['y_mean_test'] == dec].shape[0])

        decs = np.array(decs) * 100.
        empirs = np.array(empirs) * 100
        counts = np.array(counts)

        fig, axs = plt.subplots(2, figsize=(8, 8))
        # fig.suptitle('Model calibration plots')
        axs[0].scatter(decs, empirs)
        axs[0].set_title('Empirical vs predicted electricity access rate')
        axs[0].set(xlabel='$E[y|x]$, expected value of posterior-predictive distribution (%)',
                   ylabel='Empirical access rate (%)')
        axs[1].scatter(decs, counts)
        axs[1].set_title('Number of samples vs predicted electricity access rate')
        axs[1].set(xlabel='$E[y|x]$, expected value of posterior-predictive distribution (%)',
                   ylabel='Number of samples')
        plt.tight_layout()
        output_path = os.path.join(output_folder_name, f'calibration_fld_{i}.pdf')
        plt.savefig(output_path)
        plt.close('all')

        print(f'saved model calibration plots for fold {i}')



def plot_prec_recall_roc(y_test, y_prob, pos_label, output_path):
    # computing metrics
    ns_probs = [0 for _ in range(len(y_test))]
    no_skill = len(y_test[y_test == 1]) / len(y_test)
    y_pred = (y_prob > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    model_f1 = f1_score(y_test, y_pred, zero_division=1)
    cm = confusion_matrix(y_test, y_pred)

    fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=1)
    ns_roc_auc = roc_auc_score(y_test, ns_probs)
    roc_auc = roc_auc_score(y_test, y_prob)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)

    prec, recall, _ = precision_recall_curve(y_test, y_prob, pos_label=pos_label)
    pr_display = PrecisionRecallDisplay(precision=prec, recall=recall)

    model_auc = auc(recall, prec)

    # plotting
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    # consuion matrix
    ConfusionMatrixDisplay(cm).plot(ax=ax1)
    ax1.set_title(f'Acc: {acc}, F1-score: {model_f1:0.3f}')
    # roc
    roc_display.plot(ax=ax2, label='Model')
    ax2.set_title(f'ROC, AUC: {roc_auc:0.3f}, No Skill AUC: {ns_roc_auc:0.3f}')
    ax2.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    ax2.legend(loc="lower right")

    # precision-recall
    pr_display.plot(ax=ax3, label='Model')
    ax3.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    ax3.set_title(f'Precision-Recall, AUC: {model_auc:0.3f}, No Skill AUC: {no_skill:0.3f}')
    ax3.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path)
