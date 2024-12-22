import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay

def avg_cv_metrics(model, df_train, predictors, cv):
    """
    Given a model, computes and returns the average of the following metrics across
    the CV splits, along with the sum of the confusion matrices: balanced accuracy,
    precision, recall, negative predictive value, and specificity.
    """

    bal_accuracy = []
    precision = []
    recall = []
    npv = []
    specificity = []
    cm = []

    for train_idxs, val_idxs in cv:
        X_tt = df_train.loc[train_idxs, predictors]
        y_tt = df_train.loc[train_idxs, 'IN_LEAGUE_NEXT']
        X_val = df_train.loc[val_idxs, predictors]
        y_val = df_train.loc[val_idxs, 'IN_LEAGUE_NEXT']

        # fit the model and get predictions
        model.fit(X_tt, y_tt)
        y_pred = model.predict(X_val)

        # get confusion matrix
        conf = confusion_matrix(y_val, y_pred)

        # grab true negatives, true positives, false positives, false negatives
        tn = conf[0,0]
        tp = conf[1,1]
        fp = conf[0,1]
        fn = conf[1,0]

        # balanced accuracy score
        bal_accuracy.append(balanced_accuracy_score(y_val, y_pred))
        # precision (positive predictive value): what % of players classified as 1's are true 1's?
        precision.append(tp/(tp+fp))
        # recall (sensitivity): what % of actual 1's are classified as 1?
        recall.append(tp/(tp+fn))
        # negative predictive value (NPV): what % of players classified as 0's are true 0's?
        npv.append(tn/(tn+fn))
        # specificity: what % of actual 0's are classified as 0?
        specificity.append(tn/(tn+fp))
        # confusion matrix
        cm.append(conf)

    return np.mean(bal_accuracy), np.mean(precision), np.mean(recall), np.mean(npv), np.mean(specificity), sum(cm)

def print_cv_metrics(model, df_train, predictors, cv):
    """
    Given a model, displays the sum of the confusion matrices and prints the
    average of the following metrics across the CV splits: balanced accuracy,
    precision, recall, negative predictive value, and specificity.
    """

    bal_accuracy, precision, recall, npv, specificity, cm = avg_cv_metrics(model, df_train, predictors, cv)

    print('Classification metrics')
    print(f'Avg. bal. accuracy:\t {bal_accuracy:.4}')
    print(f'Avg. precision:\t\t {precision:.4}')
    print(f'Avg. recall:\t\t {recall:.4}')
    print(f'Avg. NPV:\t\t {npv:.4}')
    print(f'Avg. specificity:\t {specificity:.4}')

    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.show()
