import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    matthews_corrcoef,
    balanced_accuracy_score
)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_metrics2(y_true, y_prob, num_classes=2):

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    # Binary
    if num_classes == 2:

        prob = y_prob.reshape(-1)
        y_pred = (prob >= 0.5).astype(int)

        if len(np.unique(y_true)) > 1:
            auroc = roc_auc_score(y_true, prob)
            auprc = average_precision_score(y_true, prob)
        else:
            auroc = np.nan
            auprc = np.nan

        return {
            "AUROC": auroc,
            "AUPRC": auprc,
            "ACC": accuracy_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "MCC": matthews_corrcoef(y_true, y_pred),
            "BAcc": balanced_accuracy_score(y_true, y_pred),
        }
    # Multi-class
    else:

        y_pred = np.argmax(y_prob, axis=1)

        # AUROC OVR
        try:
            auroc = roc_auc_score(
                y_true,
                y_prob,
                multi_class="ovr"
            )
        except:
            auroc = np.nan

        return {
            "AUROC": auroc,
            "ACC": accuracy_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred, average="macro"),
            "MCC": matthews_corrcoef(y_true, y_pred),
            "BAcc": balanced_accuracy_score(y_true, y_pred),
        }


def compute_metrics(y_true, y_pred, num_classes=2):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Binary classification
    if num_classes == 2:

        prob = 1 / (1 + np.exp(-y_pred))
        y_bin = (prob >= 0.5).astype(int)

        if len(np.unique(y_true)) > 1:
            auroc = roc_auc_score(y_true, prob)
            auprc = average_precision_score(y_true, prob)
        else:
            auroc = np.nan
            auprc = np.nan

        metrics = {
            "AUROC": auroc,
            "AUPRC": auprc,
            "ACC": accuracy_score(y_true, y_bin),
            "F1": f1_score(y_true, y_bin, zero_division=0),
            "MCC": matthews_corrcoef(y_true, y_bin),
            "BAcc": balanced_accuracy_score(y_true, y_bin),
        }

        return metrics

    # Multi-class classification
    else:

        metrics = {
            "ACC": accuracy_score(y_true, y_pred),
            "Macro_F1": f1_score(y_true, y_pred, average="macro"),
            "MCC": matthews_corrcoef(y_true, y_pred),
            "BAcc": balanced_accuracy_score(y_true, y_pred),
        }

        return metrics
