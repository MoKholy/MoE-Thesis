# edited from https://github.com/othmbela/dbn-based-nids

from sklearn.metrics import classification_report, precision_recall_curve, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os


def plot_confusion_matrix(y_true, y_pred, labels, save=False, save_dir=None, filename=None):
    """Plot normalised confusion matrix"""

    confusion_mtx = confusion_matrix(y_true, y_pred)
    precision_confusion_mtx = confusion_mtx.T / (confusion_mtx.sum(axis=1)).T
    recall_confusion_mtx = confusion_mtx / confusion_mtx.sum(axis=0)

    fig = plt.figure(figsize=(21, 6))

    plt.subplot(1, 3, 1)
    _ = sns.heatmap(confusion_mtx, annot=True, cmap="Blues", fmt="", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")

    plt.subplot(1, 3, 2)
    _ = sns.heatmap(precision_confusion_mtx, annot=True, cmap="Blues", fmt='.3f', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Precision Matrix")

    plt.subplot(1, 3, 3)
    _ = sns.heatmap(recall_confusion_mtx, annot=True, cmap="Blues", fmt='.3f', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Recall Matrix")

    fig.tight_layout()
    
    if save:
        fig.savefig(os.path.join(save_dir, filename))


def plot_roc_curve(y_test, y_score, labels, save=False, avg_mode = "macro", save_dir=None, filename=None):

    n_classes = y_score.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr[avg_mode], tpr[avg_mode], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc[avg_mode] = auc(fpr[avg_mode], tpr[avg_mode])

    # Plot ROC curve
    fig = plt.figure(figsize=(14, 10))
    plt.plot(fpr[avg_mode], tpr[avg_mode], label='micro-average ROC curve (area = {0:0.4f})'.format(roc_auc[avg_mode]))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.4f})'.format(labels[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.grid()

    fig.tight_layout()

    if save:
        fig.savefig(os.path.join(save_dir, filename))


def plot_precision_recall_curve(y_test, y_score, labels, save=False, save_dir=None, filename=None):

    n_classes = y_score.shape[1]

    precision = dict()
    recall = dict()

    # Plot ROC curve
    fig = plt.figure(figsize=(14, 10))
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
        plt.plot(recall[i], precision[i], lw=2, label='Precision-Recall for {} class)'.format(labels[i]))

    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs. Recall curve')
    plt.legend(loc="best")
    plt.grid()
    
    fig.tight_layout()
    
    if save:
        fig.savefig(os.path.join(save_dir, filename))
        
        
def plot_classification_report(y_test, y_pred, labels, save=False, save_dir=None, filename=None):
    """Plot classification report for each class"""

    report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)

    fig = plt.figure(figsize=(14, 10))
    _ = sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, cmap="Blues", fmt='.3f')
    plt.xlabel("Metrics")
    plt.ylabel("Class")
    plt.title("Classification Report")

    fig.tight_layout()
    
    if save:
        fig.savefig(os.path.join(save_dir, filename))
        
def plot_train_validation_loss(train_loss, validation_loss, save=False, save_dir=None, filename=None):
    """Plot train and validation loss"""

    fig = plt.figure(figsize=(14, 10))
    plt.plot(train_loss, label='Train loss')
    plt.plot(validation_loss, label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and validation loss')
    plt.legend(loc="best")
    plt.grid()

    fig.tight_layout()
    
    if save:
        fig.savefig(os.path.join(save_dir, filename))
        
def plot_all_losses(train_total_loss, val_total_loss, train_gating_loss, val_gating_loss, train_expert_loss, val_expert_loss, save=False, save_dir=None, filename=None):
    """Plot train and validation loss"""

    fig = plt.figure(figsize=(14, 10))
    plt.plot(train_total_loss, label='Train total loss')
    plt.plot(val_total_loss, label='Validation total loss')
    plt.plot(train_gating_loss, label='Train gating loss')
    plt.plot(val_gating_loss, label='Validation gating loss')
    plt.plot(train_expert_loss, label='Train expert loss')
    plt.plot(val_expert_loss, label='Validation expert loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and validation loss')
    plt.legend(loc="best")
    plt.grid()

    fig.tight_layout()
    
    if save:
        fig.savefig(os.path.join(save_dir, filename))