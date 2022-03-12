"""Create contextualized word embeddings."""
from statistics import mode
from typing import List,Any

from torch import embedding
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from glove_vector import GloVe
import numpy as np
transformer_models = ['bert-base-uncased',
          'sentence-transformers/stsb-roberta-large',
]

class modelEmbeddings:
    def __init__(self, model_type:str = 'bert', load_direct= False)-> None:
        """Inititalize model embeddings.
        
        Parameters
        ----------
        model_type: str
            Type of transformer model to use
            Options: bert, roberta, elmo
        """
        self.model_type = model_type
        self.load_direct = load_direct

    def get_tfidf_embeddings(self, documents: List[str],save_path='models/'):
        vectorizer = TfidfVectorizer()
        embeddings = vectorizer.fit_transform(documents)
        pickle.dump(vectorizer,open(save_path+'tfidf_vectorizer.pkl','wb'))
        print("TF-IDF vectorizer saved at : models/tfidf_vectorizer.pkl")
        return embeddings

    def get_tfidf_embeddings_pre_trained(self,documents: List[str],model_path):
        vectorizer = pickle.load(open(model_path,'rb'))
        embeddings = vectorizer.transform(documents)

        return embeddings

    def __call__(self, documents: List[str],model_path = None) -> Any:
        """Output contextualized word embeddings.

        Parameters
        ----------
        documents: List[str]
            text to create embeddings
        
        Returns
        -------
        embeddings: torch.Tensor
            Word embeddings
        """
        if self.model_type =='tfidf':
            if model_path is None:
                embeddings = self.get_tfidf_embeddings(documents)
            else:
                embeddings = self.get_tfidf_embeddings_pre_trained(documents,model_path=model_path)
        elif self.model_type == 'sentence_transformer':
            if model_path is None:
                model = SentenceTransformer('bert-base-uncased')
            else:
                model = SentenceTransformer(model_path)
            embeddings = model.encode(documents)
        elif self.model_type == 'glove':
            glove_obj = GloVe()
            embeddings = np.vstack([glove_obj.create_glove_vector(doc) for doc in documents])
        else:
            embeddings = None
            
        return embeddings


if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv('data/sample.csv')
    model_embeddings = modelEmbeddings('glove')
    embeddings = model_embeddings(df['text'])
    print(embeddings)

