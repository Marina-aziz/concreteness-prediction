# concreteness_predictor.py

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import logging

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ConcretenessPredictor:
    """
    Predicts concreteness scores for words using a K-Nearest Neighbors regressor
    over distributional word embeddings.

    This class takes a word embedding loader (must implement get_vector(word)), and
    trains a KNN regressor using cosine distance. Scores are averaged (uniform or
    distance-weighted) over k nearest neighbors in embedding space.

    The embedding matrix for the training vocabulary is built and cached at fit time,
    making subsequent predictions efficient and avoiding repeated embedding lookups.
    """

    def __init__(self, embeddings, k: int = 30, weighting: str = 'uniform'):
        """
        Initialize the concreteness predictor.

        Parameters:
            embeddings: An embedding loader object (e.g., FastTextEmbeddings, TransformerStaticEmbeddings)
                that provides get_vector(word).
            k (int): Number of neighbors for KNN regression (default: 30).
            weighting (str): 'uniform' for mean, 'distance' for distance-weighted average (default: 'uniform').
        """
        self.embeddings = embeddings
        self.k = k
        self.weighting = weighting
        self.model = None
        logging.info("Initialized ConcretenessPredictor with k=%d and weighting=%s.", self.k, self.weighting)

    def _prepare_features(self, data, word_column: str = 'word') -> np.ndarray:
        """
        Converts a DataFrame of words into a matrix of embedding vectors.

        Parameters:
            data (pd.DataFrame): DataFrame with a column of words.
            word_column (str): Column name for words (default: 'word').

        Returns:
            np.ndarray: 2D array where each row is the embedding for a word.
        """
        # Stack embedding vectors for each word into a matrix
        return np.vstack([self.embeddings.get_vector(w) for w in data[word_column]])

    def fit(self, train_df, word_column: str = 'word', target_column: str = 'concreteness'):
        """
        Fit the KNN regressor on the training words and scores.

        Embeddings are cached for all training words. Model is trained using cosine
        distance and the specified weighting.

        Parameters:
            train_df (pd.DataFrame): DataFrame with columns for words and concreteness scores.
            word_column (str): Name of column with words (default: 'word').
            target_column (str): Name of column with scores (default: 'concreteness').

        Returns:
            self
        """
        # Store the words and ground-truth targets
        self.words_ = train_df[word_column].tolist()
        self.y_ = train_df[target_column].values

        # Precompute and cache embedding matrix for all training words
        self.X_emb_ = np.vstack([
            self.embeddings.get_vector(word)
            for word in self.words_
        ])

        # Build and fit the KNN regressor on embeddings
        self.model = KNeighborsRegressor(
            n_neighbors=self.k,
            metric='cosine',
            weights=self.weighting
        )
        self.model.fit(self.X_emb_, self.y_)
        logging.info("Model fitted on %d training samples (embeddings cached).", len(self.words_))
        return self

    def predict(self, data, word_column: str = 'word') -> np.ndarray:
        """
        Predict concreteness scores for new words.

        Embeddings for query words are computed (using the loader's internal cache
        where possible), and KNN regression is used to predict their scores.

        Parameters:
            data (pd.DataFrame): DataFrame containing words for prediction.
            word_column (str): Column with words to predict (default: 'word').

        Returns:
            np.ndarray: Predicted concreteness scores, one per input row.
        """
        # Stack embedding vectors for all query words
        Xq = np.vstack([self.embeddings.get_vector(w) for w in data[word_column]])
        predictions = self.model.predict(Xq)
        logging.info("Predicted concreteness scores for %d samples.", len(data))
        return predictions
