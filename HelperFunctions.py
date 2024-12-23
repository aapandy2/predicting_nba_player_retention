import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, precision_score, 
        recall_score, confusion_matrix, ConfusionMatrixDisplay,
        brier_score_loss)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.calibration import calibration_curve, CalibrationDisplay, CalibratedClassifierCV
from sklearn.model_selection import train_test_split

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
    precision, recall, negative predictive value (NPV), and specificity.
    Returns a list with balanced accuracy, precision, recall, NPV, and
    the specificity.
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

    #return single-number metrics in a list
    return [bal_accuracy, precision, recall, npv, specificity]

def ImputeAndScale(data):
    '''Impute missing data.  Fill stats data with zeros and salary with the
       mean salary for the given season.  Apply custom StandardScaler to
       normalize all numerical columns with respect to each season.'''

    # fill null values with 0 for all columns except SALARY
    null_cols = ['PER', 'TS_PERCENT', 'X3P_AR', 'F_TR', 'ORB_PERCENT',
                 'DRB_PERCENT', 'TRB_PERCENT', 'AST_PERCENT', 'STL_PERCENT',
                 'BLK_PERCENT', 'TOV_PERCENT', 'USG_PERCENT', 'WS_48']

    data[null_cols] = data[null_cols].fillna(0)
    
    mean_imputer = SimpleImputer(strategy='mean')
    
    # replace salaries of 0 with null
    data.loc[data['SALARY']==0, 'SALARY'] = None
    
    # apply mean imputer for SALARY within each season
    data['SALARY'] = (
        data
        .groupby('SEASON_START')['SALARY']
        .transform(lambda x: mean_imputer.fit_transform(x.values.reshape(-1,1)).ravel())
    )
    
    # rescale stats and salary columns within each season
    cols_to_rescale = data.select_dtypes(include=['float']).columns
    
    scaler = StandardScaler()
    
    # apply standard scaler within each season
    data[cols_to_rescale] = (
        data
        .groupby('SEASON_START')[cols_to_rescale]
        .transform(lambda x: scaler.fit_transform(x.values.reshape(-1,1)).ravel())
    )

    return data

def display_calibration(model, df, predictors, model_name):
    '''Plots a calibration curve for the provided model along with a
       histogram of predicted probabilities.  Prints the Brier score.'''
    X_test = df[predictors]
    y_test = df.IN_LEAGUE_NEXT
    
    y_prob = model.predict_proba(X_test)[:, 1]
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    
    fig, ax = plt.subplots(1, 2)
    fig.set_figheight(5)
    fig.set_figwidth(10)
    
    display = CalibrationDisplay.from_estimator(
            model,
            X_test,
            y_test,
            n_bins=10,
            name=model_name,
            ax=ax[0])
    
    ax[1].hist(y_prob)
    ax[1].set_xlabel("Predicted probability")
    ax[1].set_ylabel("Count")
    
    plt.tight_layout()
    plt.show()

    brier = brier_score_loss(y_test, y_prob)
    print(f"Brier score: {brier}")

    return None

def print_summary(best_metrics, best_hyp):
    '''Take dictionaries of best cross-validation metrics and best
       hyperparameters and return a DataFrame summarizing the info.'''
    models     = []
    bal_accs   = []
    precisions = []
    recalls    = []
    npvs       = []
    specs      = []
    hyps       = []
    
    formatted_hyps = {}
    for key in best_hyp:
        formatted_hyps[key] = []
        for param in best_hyp[key].keys():
            param_val = param
            if "__" in param:
                param_val = param.split("__")[1]
            formatted_hyps[key].append(param_val+"="+str(best_hyp[key][param]))
    
    
    for key in best_metrics:
        models.append(key)
        bal_accs.append(best_metrics[key][0])
        precisions.append(best_metrics[key][1])
        recalls.append(best_metrics[key][2])
        npvs.append(best_metrics[key][3])
        specs.append(best_metrics[key][4])
        hyps.append(formatted_hyps[key])
    
    
    results_df = pd.DataFrame({"Model":models, "Balanced accuracy":bal_accs,
                               "Precision":precisions, "Recall":recalls,
                               "NPV":npvs, "Specificity":specs,
                               "Hyperparameters":hyps})
    
    return results_df.sort_values(by="Balanced accuracy",
                                  ascending=False).reset_index(drop=True)

def npv_score(y_true, y_pred):
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    return tn / (tn + fn) if (tn + fn) > 0 else 0.0

def specificity_score(y_true, y_pred):
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

def test_model_performance(model, df_train, df_test, features):
    """
    Given a model, computes the following metrics for each test split:
    balanced accuracy, precision, recall, negative predictive value, and
    specificity. Also displays the sum of the confusion matrices and returns a
    summary DataFrame.
    """

    test_years   = []
    bal_accuracy = []
    precision = []
    recall = []
    npv = []
    specificity = []
    cm = []
    brier_uncal = []
    brier_cal   = []

    for test_year in range(2017, 2023):
        X_train = pd.concat([df_train[features],
                             df_test.loc[df_test['SEASON_START'] < test_year, features]])
        y_train = pd.concat([df_train['IN_LEAGUE_NEXT'],
                             df_test.loc[df_test['SEASON_START'] < test_year, 'IN_LEAGUE_NEXT']])

        X_tt, X_cal, y_tt, y_cal = train_test_split(X_train, y_train, 
                                                    test_size=0.2, 
                                                    random_state=23, 
                                                    shuffle=True, 
                                                    stratify=y_train)

        X_test = df_test.loc[df_test['SEASON_START'] == test_year, features]
        y_test = df_test.loc[df_test['SEASON_START'] == test_year, 'IN_LEAGUE_NEXT']

        # fit the model and get predictions, uncalibrated probabilities
        model.fit(X_tt, y_tt)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # fit calibrator and get calibrated probabilities
        model_cal = CalibratedClassifierCV(model, cv="prefit")
        model_cal.fit(X_cal, y_cal)
        y_prob_cal = model_cal.predict_proba(X_test)[:, 1]

        # get confusion matrix
        conf = confusion_matrix(y_test, y_pred)

        # save and print metrics
        test_years.append(test_year)
        bal_accuracy.append(balanced_accuracy_score(y_test, y_pred))
        precision.append(precision_score(y_test, y_pred))
        recall.append(recall_score(y_test, y_pred))
        npv.append(npv_score(y_test, y_pred))
        specificity.append(specificity_score(y_test, y_pred))
        cm.append(conf)
        brier_uncal.append(brier_score_loss(y_test, y_prob))
        brier_cal.append(brier_score_loss(y_test, y_prob_cal))

    # package metrics into a summary DataFrame
    sum_df = pd.DataFrame({"Test Year":test_years, "Balanced accuracy":bal_accuracy,
                           "Precision":precision, "Recall":recall, "NPV":npv,
                           "Specificity":specificity, "Uncalibrated Brier":brier_uncal,
                           "Calibrated Brier":brier_cal})

    # display total confusion matrix
    ConfusionMatrixDisplay(confusion_matrix=sum(cm)).plot()
    plt.title('Total confusion matrix')
    plt.show()

    return sum_df
