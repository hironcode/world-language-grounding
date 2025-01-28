import torch
import numpy as np
from scipy import stats
from sklearn import metrics

def pairwise_distance(true, predicted):
    true = torch.tensor(true, dtype=torch.float32)
    predicted = torch.tensor(predicted, dtype=torch.float32)

    dlat = true[:, None, 0] - predicted[None, :, 0]
    dlon = true[:, None, 1] - predicted[None, :, 1]

    return np.sqrt(dlat**2 + dlon**2).numpy()

def pairwise_abs_distance_fn(true_values, predicted_values):
    true_values = true_values.ravel()
    return np.abs(true_values[:, np.newaxis] - predicted_values[:, np.newaxis].T)

def compute_proximity_error_matrix(true_values, predicted_values, is_place):
    if len(true_values) != len(predicted_values):
        raise ValueError("Input arrays must have the same length")

    if is_place:
        dist_matrix = pairwise_distance(true_values, predicted_values)
    else:
        dist_matrix = pairwise_abs_distance_fn(true_values, predicted_values)
    # 少なくとも二、三次元じゃないといけない
    target_diff = np.diag(dist_matrix)
    error_matrix = dist_matrix < target_diff[:, np.newaxis]
    return error_matrix

def proximity_scores(error_matrix, is_test):
    train_error = error_matrix[~is_test, :][:, ~is_test].mean(axis=1)
    test_error = error_matrix[is_test, :][:, is_test].mean(axis=1)
    combined_error = error_matrix.mean(axis=1)
    return train_error, test_error, combined_error

def score_place_probe(target, pred):
    x_pearson = stats.pearsonr(target[:, 0], pred[:, 0])
    x_spearman = stats.spearmanr(target[:, 0], pred[:, 0])
    x_kendall = stats.kendalltau(target[:, 0], pred[:, 0])

    y_pearson = stats.pearsonr(target[:, 1], pred[:, 1])
    y_spearman = stats.spearmanr(target[:, 1], pred[:, 1])
    y_kendall = stats.kendalltau(target[:, 1], pred[:, 1])

    score_dict = {
        'x_r2': metrics.r2_score(target[:, 0], pred[:, 0]),
        'y_r2': metrics.r2_score(target[:, 1], pred[:, 1]),
        'r2': metrics.r2_score(target, pred),
        'x_mae': metrics.mean_absolute_error(target[:, 0], pred[:, 0]),
        'y_mae': metrics.mean_absolute_error(target[:, 1], pred[:, 1]),
        'mae': metrics.mean_absolute_error(target, pred),
        'mse': metrics.mean_squared_error(target, pred),
        'rmse': np.sqrt(metrics.mean_squared_error(target, pred)),
        'x_pearson': x_pearson.correlation,
        'x_pearson_p': x_pearson.pvalue,
        'x_spearman': x_spearman.correlation,
        'x_spearman_p': x_spearman.pvalue,
        'x_kendall': x_kendall.correlation,
        'x_kendall_p': x_kendall.pvalue,
        'y_pearson': y_pearson.correlation,
        'y_pearson_p': y_pearson.pvalue,
        'y_spearman': y_spearman.correlation,
        'y_spearman_p': y_spearman.pvalue,
        'y_kendall': y_kendall.correlation,
        'y_kendall_p': y_kendall.pvalue
    }
    return score_dict

def score_time_probe(target, pred):
    target = target.ravel()
    pearson = stats.pearsonr(target, pred)
    spearman = stats.spearmanr(target, pred)
    kendall = stats.kendalltau(target, pred)
    score_dict = {
        'mae': metrics.mean_absolute_error(target, pred),
        'mse': metrics.mean_squared_error(target, pred),
        'rmse': np.sqrt(metrics.mean_squared_error(target, pred)),
        'r2': metrics.r2_score(target, pred),
        'pearson': pearson.correlation,
        'pearson_p': pearson.pvalue,
        'spearman': spearman.correlation,
        'spearman_p': spearman.pvalue,
        'kendall': kendall.correlation,
        'kendall_p': kendall.pvalue
    }
    return score_dict
