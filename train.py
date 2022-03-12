"""Train machine learning classifier."""
from typing import Any, List
import pickle
from torch import embedding
from model_embeddings import modelEmbeddings
from sklearn import svm
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import classification_report,accuracy_score


class Classifier:
    def __init__(self,dataframe:pd.DataFrame,embedding_type, model_path=None)-> None:
        """Inititalize classifier."""
        self.embeddings_model = modelEmbeddings(embedding_type,model_path)
        self.create_split(dataframe)
        
    def create_split(self,dataframe):
        """Return train/test embeddings and class"""

        train_df,test_df = train_test_split(dataframe,test_size=0.1)
        
        self.train_df = train_df
        self.test_df = test_df
        self.x_train = self.embeddings_model(train_df['text'].values)
        self.y_train = train_df['Label']
        
        self.x_test = self.embeddings_model(test_df['text'].values)
        self.y_test = test_df['Label']

        

    def get_train_test_split(self):
        return self.x_train,self.x_test,self.y_train,self.y_test
    
    def train_svm(self,)->None:
        """Train model.
        
        Parameters
        ----------
        model: Any

        """
        
        X_train, X_test, y_train, y_test = self.get_train_test_split()

        self.svm_model = svm.SVC(kernel='linear', C=3).fit(X_train, y_train)
        y_pred = self.svm_model.predict(X_test)
        print('Accuracy: SVM model = '+str(round(accuracy_score(y_test,y_pred)*100,2)))
        print(classification_report(y_test,y_pred))

        pickle.dump(self.svm_model,open('models/svm.pkl','wb'))

if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv('data/sample.csv')
    print(df.head())
    classifier=Classifier(df,embedding_type= 'sentence_transformer',model_path ='sentence-transformers/stsb-roberta-large' )
    classifier.train_svm()
