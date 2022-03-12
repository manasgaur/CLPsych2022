"""Calculate GloVe vector for input string."""
import codecs
import logging
import re
from pathlib import Path
from typing import List

import numpy as np



class GloVe:
    """Class to process transform string into word vectors."""

    def __init__(
        self, embeddings_path: Path = "models/glove.6B.300d.txt", embeddings_size: int = 300
    ) -> None:
        """Initialize GloVe class."""
        # Create dict to store pre-trained vectors.
        self.embeddings_path = embeddings_path
        self.embedding_size = embeddings_size
        self.word_regex = re.compile(r"[a-zA-Z]+")

        self.embedding_dict = {}
        with codecs.open(self.embeddings_path, "r", "utf-8") as f:
            # Iterate over lines, each line contains 'word' followed by it's vector
            for line in f:
                values = line.split()
                word = values[0]
                vectors = np.asarray(values[1:], "float64")
                self.embedding_dict[word] = vectors

    def calculate_mean_vector(self, list_of_vectors: List) -> np.ndarray:
        """Calculate final vector using numpy.
        Parameters
        ----------
        list_of_vectors : List
            List of GloVe vectors
        Returns
        -------
        vector : numpy.ndarray
            vector of 100-dimension
        """
        # Check if list_of_vectors is empty
        if not list_of_vectors:
            # Create zero vector
            vector = np.zeros(self.embedding_size).astype("float64")
        else:
            # Calculate mean of vectors
            array_of_vectors = np.array(list_of_vectors).astype("float64")
            vector = np.sum(array_of_vectors, axis=0)
            vector = vector / np.sqrt((vector ** 2).sum())
        return vector

    def create_glove_vector(self, text: str) -> np.ndarray:
        """Convert input_text/headers to GloVe vector.
        Parameters
        ----------
        text : str
            text/headers to vectorize
        Returns
        -------
        vector : numpy.ndarray
            vector of 100-dimension
        """
        # Shortlist words without digits and special charachters.
        words = re.findall(self.word_regex, text)
        list_of_vectors = []
        # Loop over words to get GloVe vector of each word.
        for word in words:
            word_lowered = word.lower()
            # Check if word is present in dictionary of vecotors or not.
            if word_lowered in self.embedding_dict:
                list_of_vectors.append(self.embedding_dict[word_lowered])

        vector = self.calculate_mean_vector(list_of_vectors)
        return vector