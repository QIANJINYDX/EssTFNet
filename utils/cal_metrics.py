from sklearn import metrics
import numpy as np

def cal_metrics(all_trues, all_scores, threshold):
    """ Calculate the evaluation metrics """
    all_preds = (all_scores >= threshold)
    acc = metrics.accuracy_score(all_trues, all_preds)
    pre = metrics.precision_score(all_trues, all_preds)
    rec = metrics.recall_score(all_trues, all_preds)
    f1 = metrics.f1_score(all_trues, all_preds)
    mcc = metrics.matthews_corrcoef(all_trues, all_preds)
    fpr, tpr, _ = metrics.roc_curve(all_trues, all_scores)
    AUC = metrics.auc(fpr, tpr)
    AUPR = metrics.average_precision_score(all_trues, all_scores)
    tn, fp, fn, tp = metrics.confusion_matrix(all_trues, all_preds, labels=[0, 1]).ravel()
    return tp, tn, fp, fn, acc, f1, pre, rec, mcc, AUC, AUPR

def cal_regression_metrics(y_true, y_pred):
    """Calculate regression metrics including MSE, RMSE, MAE, R2, and Adjusted R2"""
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    
    # Calculate adjusted R2
    n = len(y_true)
    p = 1  # number of predictors
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Adjusted_R2': adjusted_r2
    }

def print_metrics(data_type, essentiality_loss, metrics):
    """ Print the evaluation results """
    tp, tn, fp, fn, acc, f1, pre, rec, mcc, auc, aupr = metrics
    res = '\t'.join([
        '%s:' % data_type,
        'TP=%-5d' % tp,
        'TN=%-5d' % tn,
        'FP=%-5d' % fp,
        'FN=%-5d' % fn,
        'essentiality_loss:%0.5f' % essentiality_loss,
        'acc:%0.5f' % acc,
        'f1:%0.5f' % f1,
        'pre:%0.5f' % pre,
        'rec:%0.5f' % rec,
        'mcc:%0.5f' % mcc,
        'auc:%0.5f' % auc,
        'aupr:%0.5f' % aupr
    ])
    return res

def print_regression_metrics(data_type, metrics_dict):
    """Print regression metrics in a formatted string"""
    res = '\t'.join([
        '%s:' % data_type,
        'MSE:%0.5f' % metrics_dict['MSE'],
        'RMSE:%0.5f' % metrics_dict['RMSE'],
        'MAE:%0.5f' % metrics_dict['MAE'],
        'R2:%0.5f' % metrics_dict['R2'],
        'Adjusted_R2:%0.5f' % metrics_dict['Adjusted_R2']
    ])
    return res

def best_f1_thr(y_true, y_score):
    """ Calculate the best threshold  with f1 """
    best_thr = 0.5
    best_f1 = 0
    for thr in range(1,100):
        thr /= 100
        tp, tn, fp, fn, acc, f1, pre, rec, mcc, AUC, AUPR = cal_metrics(y_true, y_score, thr)
        if f1>best_f1:
            best_f1 = f1
            best_thr = thr
    return best_thr, best_f1

def best_acc_thr(y_true, y_score):
    """ Calculate the best threshold with acc """
    best_thr = 0.5
    best_acc = 0
    for thr in range(1,100):
        thr /= 100
        tp, tn, fp, fn, acc, f1, pre, rec, mcc, AUC, AUPR = cal_metrics(y_true, y_score, thr)
        if acc>best_acc:
            best_acc = acc
            best_thr = thr
    return best_thr, best_acc