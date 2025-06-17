# embedding_loader.py

from abc import ABC, abstractmethod
import numpy as np
import fasttext
from gensim.models import KeyedVectors
import os
import logging

# Set up logging for the module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BaseEmbeddings(ABC):
    """
    Abstract base class for word embeddings.

    Specifies the required interface for embedding loaders.
    All embedding classes must implement load_embeddings() and get_vector().
    """

    @abstractmethod
    def load_embeddings(self):
        """
        Load embeddings from the specified source.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def get_vector(self, word: str) -> np.ndarray:
        """
        Retrieve the embedding vector for the given word.
        Must be implemented by subclasses.

        Parameters:
            word (str): The target word.

        Returns:
            np.ndarray: The word's embedding vector.
        """
        pass

class FastTextEmbeddings(BaseEmbeddings):
    """
    Embedding loader for fastText models (.bin or .vec).

    Supports:
        - Binary (.bin): OOV vector computation using subword info.
        - Text (.vec): Precomputed vectors only, no subword support.
    Provides vector caching for efficiency.
    """

    def __init__(self, model_path: str, file_format: str = 'bin'):
        """
        Initialize a FastTextEmbeddings loader.

        Parameters:
            model_path (str): Path to the fastText model file.
            file_format (str): 'bin' for binary, 'vec' for text format (default 'bin').
        """
        self.model_path = model_path
        self.file_format = file_format.lower()
        self.model = None
        self._vector_cache: dict[str, np.ndarray] = {}  # Local cache for fast vector lookup

    def load_embeddings(self):
        """
        Load the fastText embedding model.

        Raises:
            FileNotFoundError: If the model file is missing.
            ValueError: For unknown file formats.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file {self.model_path} not found.")

        if self.file_format == 'bin':
            # Use fasttext package for binary models (supports subword OOVs)
            self.model = fasttext.load_model(self.model_path)
            logging.info("Loaded fastText binary model from %s with %d dimensions.",
                         self.model_path, self.model.get_dimension())
        elif self.file_format == 'vec':
            # Use gensim for .vec (plain text word vectors)
            self.model = KeyedVectors.load_word2vec_format(self.model_path, binary=False)
            logging.info("Loaded fastText vector file from %s with %d dimensions.",
                         self.model_path, self.model.vector_size)
        else:
            raise ValueError("Invalid file_format. Please use 'bin' or 'vec'.")

    def get_vector(self, word: str) -> np.ndarray:
        """
        Return the embedding vector for a word.

        - For binary models, returns OOV vectors using subword information.
        - For .vec models, returns zero vector if OOV.
        - Uses caching for efficiency on repeated lookups.

        Parameters:
            word (str): Word to look up.

        Returns:
            np.ndarray: Word embedding.

        Raises:
            ValueError: If model not loaded.
        """
        if self.model is None:
            logging.error("Embeddings not loaded. Call load_embeddings() first.")
            raise ValueError("Embeddings not loaded. Call load_embeddings() first.")

        # Check cache before computing
        if word in self._vector_cache:
            return self._vector_cache[word]

        if self.file_format == 'bin':
            # .bin model always returns a vector (with subword support)
            vec = self.model.get_word_vector(word)
        else:
            # .vec model: OOV handled by returning zeros
            vec = self.model[word] if word in self.model else np.zeros(self.model.vector_size)
        self._vector_cache[word] = vec
        return vec

class TransformerStaticEmbeddings(BaseEmbeddings):
    """
    Loader for static embeddings distilled from transformer models (e.g., BERT, RoBERTa, GPT2).

    Loads precomputed embeddings in word2vec text format, typically produced by
    a distillation process (Gupta & Jaggi, 2021). Caches lookups for efficiency.
    """

    def __init__(self, model_path: str):
        """
        Initialize the TransformerStaticEmbeddings loader.

        Parameters:
            model_path (str): Path to the static embeddings file (.txt or .vec).
        """
        self.model_path = model_path
        self.model = None
        self._vector_cache: dict[str, np.ndarray] = {}

    def load_embeddings(self):
        """
        Load the transformer static embeddings using gensim.

        Raises:
            FileNotFoundError: If file is missing.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file {self.model_path} not found.")
        self.model = KeyedVectors.load_word2vec_format(self.model_path, binary=False)
        logging.info("Loaded transformer static embeddings from %s with dimension %d",
                     self.model_path, self.model.vector_size)

    def get_vector(self, word: str) -> np.ndarray:
        """
        Retrieve the embedding vector for a word.

        - Returns a zero vector if word not in vocabulary.
        - Uses cache for repeated lookups.

        Parameters:
            word (str): Word to look up.

        Returns:
            np.ndarray: Word embedding, or zero vector if OOV.

        Raises:
            ValueError: If embeddings are not loaded.
        """
        if self.model is None:
            raise ValueError("Embeddings not loaded. Call load_embeddings() first.")

        # Check cache first
        if word in self._vector_cache:
            return self._vector_cache[word]

        # OOV handling: zero vector for missing words
        vec = self.model[word] if word in self.model else np.zeros(self.model.vector_size)
        self._vector_cache[word] = vec
        return vec
