"""Create contextualized word embeddings."""
from json import load
from statistics import mode
from typing import List,Any

from torch import embedding
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from glove_vector import GloVe
import numpy as np
from scipy import sparse
transformer_models = ['bert-base-uncased',
          'sentence-transformers/stsb-roberta-large',
]

class modelEmbeddings:
    def __init__(self, model_type:str = 'bert')-> None:
        """Inititalize model embeddings.
        
        Parameters
        ----------
        model_type: str
            Type of transformer model to use
            Options: bert, roberta, elmo
        save_path: str
            save location of embeddings model
        load_path: str
            location of pre-trained embeddings model
        """
        self.model_type = model_type
        self.vectorizer_path = None

    def get_tfidf_embeddings(self, documents: List[str], save_path: str)->sparse:
        """Train tf-idf vectorizer.
       
        Parameters
        ----------
        documents: List[str]
            List of text documents
        save_path: str
            saving path to tf-idf vectorizer

        Returns
        -------
        embeddings: sprase
            Sparse matrix containing tf-idf vectors
        """
        vectorizer = TfidfVectorizer()
        embeddings = vectorizer.fit_transform(documents)
        self.vectorizer_path = save_path+'tfidf_vectorizer.pkl'

        pickle.dump(vectorizer,open(self.vectorizer_path,'wb'))
        print("TF-IDF vectorizer saved at : models/tfidf_vectorizer.pkl")
        return embeddings

    def get_tfidf_embeddings_pre_trained(self, documents: List[str], load_path: str)->sparse:
        """Use pre-trained vectorizer.
        
        Parameters
        ----------
        documents: List[str]
            List of text documents
        load_path: str
            path to pre-trained tf-idf vectorizer

        Returns
        -------
        embeddings: sprase
            Sparse matrix containing tf-idf vectors
        """
        vectorizer = pickle.load(open(load_path,'rb'))
        embeddings = vectorizer.transform(documents)

        return embeddings

    def __call__(self, documents: List[str], load_path = None, save_path = 'models/') -> Any:
        """Output contextualized word embeddings.

        Parameters
        ----------
        documents: List[str]
            text to create embeddings
        load_path: str
            path to pre-trained tf-idf vectorizer 
        save_path: str
            saving path to tf-idf vectorizer
        
        Returns
        -------
        embeddings: Any
            Word embeddings
        self.vectorizer_path: str
            Path to trained model
        """
        if self.model_type =='tfidf':
            if load_path is None:
                embeddings = self.get_tfidf_embeddings(documents,save_path=save_path)
            else:
                embeddings = self.get_tfidf_embeddings_pre_trained(documents,load_path=load_path)
        elif self.model_type == 'sentence_transformer':
            if load_path is None:
                model = SentenceTransformer('bert-base-uncased')
            else:
                model = SentenceTransformer(load_path)
            embeddings = model.encode(documents)
        elif self.model_type == 'glove':
            glove_obj = GloVe()
            embeddings = np.vstack([glove_obj.create_glove_vector(doc) for doc in documents])
        else:
            embeddings = None
            
        return embeddings, self.vectorizer_path


if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv('data/sample.csv')
    model_embeddings = modelEmbeddings('tfidf',)
    embeddings = model_embeddings(df['text'])
    print(embeddings)

