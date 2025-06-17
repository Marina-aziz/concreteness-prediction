# data_processing.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import logging

# Configure logging for the entire module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataProcessor:
    """
    A class to handle data loading, splitting, and creating cross-validation folds
    for the concreteness prediction task.
    """

    def __init__(self, file_path: str, word_column: str = 'word', score_column: str = 'concreteness',
                 train_file: str = 'train_set.csv', test_file: str = 'test_set.csv', column_mapping: list = None):
        """
        Initialize the DataProcessor.

        Parameters:
            file_path (str): Path to the dataset file (Excel or CSV).
            word_column (str): Name of the column containing the words.
            score_column (str): Name of the column containing concreteness scores.
            train_file (str): Filename for the training set CSV.
            test_file (str): Filename for the test set CSV.
            column_mapping (list): Optional list of column names to rename the loaded dataset.
        """
        self.file_path = file_path
        data_dir = os.path.dirname(file_path)
        self.train_file = os.path.join(data_dir, train_file)
        self.test_file  = os.path.join(data_dir, test_file)
        self.word_column = word_column
        self.score_column = score_column
        self.column_mapping = column_mapping
        self.data = None
        self.train_set = None
        self.test_set = None

    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset from the specified file. Supports both CSV and Excel formats.
        Adjusts column headers after loading if column_mapping is provided.
        
        Returns:
            pd.DataFrame: The loaded dataset.
        """
        # Load Excel or CSV depending on file extension
        if self.file_path.endswith('.xlsx'):
            self.data = pd.read_excel(self.file_path)
        else:
            self.data = pd.read_csv(self.file_path)
        
        # Adjust column headers to match expected format if column_mapping is provided.
        if self.column_mapping is not None:
            self.data.columns = self.column_mapping
        else:
            # Default mapping if not provided; expected columns are set explicitly
            self.data.columns = ['word', 'col2', 'concreteness', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9', 'col10', 'frequency']
        
        logging.info("Data loaded successfully from %s", self.file_path)
        return self.data

    def _create_bins(self, data: pd.DataFrame) -> pd.Series:
        """
        Create equal-frequency bins for stratified splitting based on concreteness scores.
        
        Parameters:
            data (pd.DataFrame): The dataset for which to create bins.
        
        Returns:
            pd.Series: A series representing binned concreteness scores.
        """
        # Use quantile-based binning (qcut) to discretize continuous scores
        return pd.qcut(data[self.score_column], q=5, labels=False)

    def split_data(self, test_size: float = 0.1, random_state: int = 42, stratify: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the dataset into training and test sets.
        Supports stratification based on binned concreteness scores if stratify=True.
        If the split files already exist, load them instead of performing a new split.
        
        Parameters:
            test_size (float): Proportion of the dataset to include in the test split (default is 0.1).
            random_state (int): Random seed for reproducibility.
            stratify (bool): Split the data based on the equal frequency binning strategy.

        Returns:
            tuple: (train_set, test_set) as pandas DataFrames.
        """
        # Check if both split datasets already exist, load them if available.
        if os.path.exists(self.train_file) and os.path.exists(self.test_file):
            logging.info("Found both training and test set files, loading data...")
            self.train_set = pd.read_csv(self.train_file)
            self.test_set = pd.read_csv(self.test_file)
            return self.train_set, self.test_set

        # Load data if not already loaded.
        if self.data is None:
            self.load_data()

        if stratify:
            # Create equal-frequency bins for stratified splitting.
            self.data['score_bin'] = self._create_bins(self.data)
            strat = self.data['score_bin']
        else:
            strat = None

        self.train_set, self.test_set = train_test_split(
            self.data,
            test_size=test_size,
            random_state=random_state,
            stratify=strat
        )
        
        # Save the split datasets to CSV files.
        self.train_set.to_csv(self.train_file, index=False)
        self.test_set.to_csv(self.test_file, index=False)

        logging.info("Data split complete. Training set saved to '%s' and test set saved to '%s'.", self.train_file, self.test_file)
        return self.train_set, self.test_set

    def create_folds(self, n_splits: int = 5, random_state: int = 42, stratify: bool = False):
        """
        Create stratified or plain k-fold splits for cross-validation using the training set.
        Uses equal-frequency bins for stratification if enabled.

        Parameters:
            n_splits (int): Number of folds (default is 5).
            random_state (int): Random seed for reproducibility.
            stratify (bool): Split the data based on the equal frequency binning strategy.

        Returns:
            generator: Yields (train_index, validation_index) for each fold based on the training set.
        """
        # Use the current training set for cross-validation folds.
        subset = self.train_set.copy()

        if stratify:
            # Create bins for stratified k-fold splitting.
            subset['score_bin'] = self._create_bins(subset)
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            logging.info("Created %d stratified folds for cross-validation.", n_splits)
            return skf.split(subset, subset['score_bin'])
        else:
            # No stratification: plain KFold splitting
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            logging.info("Created %d plain folds for cross-validation (no stratification).", n_splits)
            return splitter.split(subset)
