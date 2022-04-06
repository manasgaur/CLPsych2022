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

from data_reader import get_train_users
transformer_models = ['bert-base-uncased',
          'sentence-transformers/stsb-roberta-large',
]

class modelEmbeddings:
    def __init__(self, model_type:str = 'bert', save_path ='models/', load_path = None)-> None:
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
        self.save_path = save_path
        self.load_path = load_path

    def get_tfidf_embeddings(self, documents: List[str],save_path='models/'):
        vectorizer = TfidfVectorizer()
        embeddings = vectorizer.fit_transform(documents) # fit_transform while training data
        pickle.dump(vectorizer,open(save_path+'tfidf_vectorizer.pkl','wb'))
        print("TF-IDF vectorizer saved at : models/tfidf_vectorizer.pkl")
        return embeddings

    def get_tfidf_embeddings_pre_trained(self,documents: List[str],load_path):
        vectorizer = pickle.load(open(load_path,'rb'))
        embeddings = vectorizer.transform(documents) #For test data use transform.

        return embeddings

    def __call__(self, documents: List[str]) -> Any:
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
            if self.load_path is None:
                embeddings = self.get_tfidf_embeddings(documents,save_path=self.save_path)
            else:
                embeddings = self.get_tfidf_embeddings_pre_trained(documents,load_path=self.load_path)
        elif self.model_type == 'sentence_transformer':
            if self.load_path is None:
                model = SentenceTransformer('bert-base-uncased')
            else:
                model = SentenceTransformer(self.load_path)
            embeddings = model.encode(documents)
        elif self.model_type == 'glove':
            glove_obj = GloVe()
            embeddings = np.vstack([glove_obj.create_glove_vector(doc) for doc in documents])
        else:
            embeddings = None
            
        return embeddings


if __name__ == '__main__':
    import pandas as pd

    # Loading the data and merging into a large pd.DataFrame
    users = get_train_users()
    dfs = []
    for user in users.keys():
        for i in range(len(users[user]['data'])):
            tdf = users[user]['data'][i]
            tdf.title = tdf.title.fillna(' ')
            tdf.content = tdf.content.fillna(' ')
            tdf['timeline_id'] = users[user]['timelines'][i]
            dfs.append(tdf)
    df = pd.concat(dfs)

    #df = pd.read_csv('data/sample.csv')
    model_embeddings = modelEmbeddings('glove')
    embeddings = model_embeddings(df['content'])
    print(embeddings)

