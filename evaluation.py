# evaluation.py

import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
import logging

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Evaluator:
    """
    Evaluates the performance of a concreteness prediction model.

    Provides methods to compute RMSE, Pearson, and Spearman correlation metrics.
    Supports both held-out test evaluation and k-fold cross-validation.
    """

    def __init__(self, predictor, word_column: str = 'word', target_column: str = 'concreteness'):
        """
        Initialize the Evaluator.

        Parameters:
            predictor: Instance of ConcretenessPredictor (must implement fit() and predict()).
            word_column (str): Name of column containing words (default: 'word').
            target_column (str): Name of column containing concreteness scores (default: 'concreteness').
        """
        self.predictor = predictor
        self.word_column = word_column
        self.target_column = target_column

    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Compute standard evaluation metrics for predicted vs. true values.

        Parameters:
            y_true (np.ndarray): True concreteness scores.
            y_pred (np.ndarray): Predicted concreteness scores.

        Returns:
            dict: {
                'rmse': Root Mean Squared Error,
                'pearson_corr': Pearson correlation coefficient,
                'pearson_p': p-value for Pearson correlation,
                'spearman_corr': Spearman correlation coefficient,
                'spearman_p': p-value for Spearman correlation
            }
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        pearson_corr, pearson_p = pearsonr(y_true, y_pred)
        spearman_corr, spearman_p = spearmanr(y_true, y_pred)
        return {
            'rmse': rmse,
            'pearson_corr': pearson_corr,
            'pearson_p': pearson_p,
            'spearman_corr': spearman_corr,
            'spearman_p': spearman_p
        }

    def evaluate_cross_validation(self, train_set, folds_generator) -> dict:
        """
        Perform k-fold cross-validation, reporting mean metrics across folds.

        Each fold trains the predictor on train split and evaluates on validation split.

        Parameters:
            train_set (pd.DataFrame): Training set with words and concreteness scores.
            folds_generator: Generator of (train_idx, val_idx) pairs, e.g. from DataProcessor.create_folds().

        Returns:
            dict: {
                'avg_rmse': Mean RMSE over all folds,
                'avg_pearson_corr': Mean Pearson correlation,
                'avg_spearman_corr': Mean Spearman correlation
            }
        """
        rmse_list = []
        pearson_list = []
        spearman_list = []

        for fold_index, (train_index, val_index) in enumerate(folds_generator):
            # Extract training and validation splits for the fold
            fold_train = train_set.iloc[train_index]
            fold_val = train_set.iloc[val_index]

            # Train predictor on training subset
            self.predictor.fit(fold_train, word_column=self.word_column, target_column=self.target_column)
            # Predict on validation subset
            y_pred = self.predictor.predict(fold_val, word_column=self.word_column)
            y_true = fold_val[self.target_column].values

            # Compute and record metrics for the fold
            metrics = self.compute_metrics(y_true, y_pred)
            rmse_list.append(metrics['rmse'])
            pearson_list.append(metrics['pearson_corr'])
            spearman_list.append(metrics['spearman_corr'])

            logging.info("Fold %d evaluation - RMSE: %.4f, Pearson Corr: %.4f (p=%.4f), Spearman Corr: %.4f (p=%.4f)",
                         fold_index + 1, metrics['rmse'], metrics['pearson_corr'], metrics['pearson_p'], metrics['spearman_corr'], metrics['spearman_p'])

        # Calculate and report averages across all folds
        avg_rmse = np.mean(rmse_list)
        avg_pearson = np.mean(pearson_list)
        avg_spearman = np.mean(spearman_list)
        logging.info("Cross-validation complete - Average RMSE: %.4f, Average Pearson Corr: %.4f, Average Spearman Corr: %.4f",
                     avg_rmse, avg_pearson, avg_spearman)
        return {
            'avg_rmse': avg_rmse,
            'avg_pearson_corr': avg_pearson,
            'avg_spearman_corr': avg_spearman
        }

    def evaluate_test_set(self, test_set) -> dict:
        """
        Evaluate predictor on a held-out test set and report metrics.

        Assumes predictor has been trained (fit) on the full training set.

        Parameters:
            test_set (pd.DataFrame): DataFrame with words and concreteness scores.

        Returns:
            dict: {
                'rmse': Root Mean Squared Error,
                'pearson_corr': Pearson correlation,
                'pearson_p': Pearson p-value,
                'spearman_corr': Spearman correlation,
                'spearman_p': Spearman p-value
            }
        """
        # Predict on test set words
        y_pred = self.predictor.predict(test_set, word_column=self.word_column)
        y_true = test_set[self.target_column].values
        metrics = self.compute_metrics(y_true, y_pred)
        logging.info("Test set evaluation - RMSE: %.4f, Pearson Corr: %.4f (p=%.4f), Spearman Corr: %.4f (p=%.4f)",
                     metrics['rmse'], metrics['pearson_corr'], metrics['pearson_p'],
                     metrics['spearman_corr'], metrics['spearman_p'])
        return metrics
