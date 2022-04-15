"""Train machine learning classifier."""
from typing import Any, List,Optional
import pickle
import numpy as np
from model_embeddings import modelEmbeddings
from sklearn import svm
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import classification_report,accuracy_score
from multiclass_bert_trainer import DistillBERTClass
from data_reader import BertDataset
from transformers import get_linear_schedule_with_warmup
from utils import TRAIN_BATCH_SIZE,EPOCHS,TOKENS_MAX_LENGTH
import torch
from multiclass_bert_trainer import train_fn,eval_fn,predict_fn
import torch
torch.zeros(1).cuda()
from data_reader import get_train_users
from transformers import DistilBertTokenizer

class Classifier:
    def __init__(self, dataframe:pd.DataFrame, embeddings_model_type: str, vectorizer_path: Optional[str] = None, save_dir: str = 'models/',eval=False) -> None:
        """Inititalize classifier.
        
        Parameters
        ----------
        dataframe: pd.DataFrame
            dataset
        embeddings_model_type: str
            type of embeddings. e.g glove/tfidf/sentence_transformers
        vectorizer_path: Optional[str]
            Path to pre-trained vectors/ transformer model.
        save_dir: str
            Save directory for ML models.
        """
        self.embeddings_model = modelEmbeddings(embeddings_model_type)
        self.vectorizer_path = vectorizer_path
        self.saved_models_path = save_dir
        self.dataframe =dataframe
        if not eval:
            self.create_split(self.dataframe)

    def create_split(self, dataframe:pd.DataFrame) -> None:
        """Return train/test embeddings and class
        
        Parameters
        ----------
        dataframe: pd.DataFrame
            dataset
        """
        train_df,test_df = train_test_split(dataframe,test_size=0.1,random_state=42, stratify=dataframe['label'].values)
        
        self.train_df = train_df
        self.test_df = test_df

        self.y_train = train_df['label']
        self.y_test = test_df['label']

        if not self.vectorizer_path:
            self.x_train,self.vectorizer_path = self.embeddings_model(train_df['content'].values,save_path=self.saved_models_path)
        else:
            self.x_train,self.vectorizer_path = self.embeddings_model(train_df['content'].values,load_path=self.vectorizer_path)
        
        self.x_test,self.vectorizer_path = self.embeddings_model(test_df['content'].values, load_path = self.vectorizer_path)

    def get_train_test_split(self) -> Any:
        """Return data split."""
        return self.x_train,self.x_test,self.y_train,self.y_test
    
    def train_predict(self,)->None:
        """Train and save model."""
        X_train, X_test, y_train, y_test = self.get_train_test_split()

        self.svm_model = svm.SVC(kernel='linear', C=3).fit(X_train, y_train)
        self.y_pred = self.svm_model.predict(X_test)
        save_loc = self.saved_models_path+'svm.pkl'

        print('Accuracy: SVM model = '+str(round(accuracy_score(y_test,self.y_pred)*100,2)))
        print(classification_report(y_test,self.y_pred))

        pickle.dump(self.svm_model,open(save_loc,'wb'))
        print("Model saved at: {}".format(save_loc))

        
    def predict(self,model_path: str)-> None:
        """Predict using trained model.
        
        Parameters
        ----------
        model_path: str
            Save directory for ML models.
        
        Returns
        -------
        pred_list: List[str]
            List of predicted labels
        test_list: List[str]
            List of true labels if any
        """
        embeddings,_ = self.embeddings_model(self.dataframe['content'])
        classifier = pickle.load(open(model_path,'rb'))
        pred_list = classifier.predict(embeddings)
        if 'label' in self.dataframe:
            test_list = self.dataframe['label']
        else:
            test_list = None
        return pred_list,test_list


class BertClassifier:
    def __init__(self,dataframe:pd.DataFrame,device='cuda',model_path=None)-> None:
        """"""
        self.dataframe = dataframe
        self.num_classes = len(dataframe['label'].unique())
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')

        self.create_split(self.dataframe)
        if model_path:
             self.modeltorch.load(model_path).to(self.device)
        else:
            self.model = DistillBERTClass()
        self.device = device
        self.model.to(self.device)

    def create_split(self,dataframe):
        """Return train/test embeddings and class"""

        train_df,dev_df = train_test_split(dataframe,test_size=0.1,random_state=42, stratify=dataframe.label.values)
        
        self.train_df = train_df
        self.dev_df = dev_df

        
        train_dataset = BertDataset(
                       self.train_df,self.tokenizer,TOKENS_MAX_LENGTH
                       )
        self.train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=TRAIN_BATCH_SIZE,
                                                        )


        valid_dataset = BertDataset(
                           self.dev_df,self.tokenizer,TOKENS_MAX_LENGTH)
        self.valid_data_loader = torch.utils.data.DataLoader(valid_dataset,
                                                        batch_size=TRAIN_BATCH_SIZE)

    def train(self):
        """Train Distilbert classifier"""
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [{
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],"weight_decay": 0.001,},
            {   "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],"weight_decay": 0.0,}]
        optimizer = torch.optim.Adam(optimizer_parameters, lr=2e-5)
        for epoch in range(EPOCHS):
            train_fn(self.train_data_loader,self.model,optimizer,epoch,device = self.device)
            final_output,final_targets,epoch_accu = eval_fn(self.model, self.valid_data_loader,device=self.device)
            print("Accuracy on test data = %0.2f%%" % epoch_accu)

            print('------------------------')

    def predict(self,test_dataframe):
        """Predict on input dataframe"""
        test_dataframe['label'] = [0]*len(test_dataframe)
        test_dataset =  BertDataset(
                           test_dataframe,self.tokenizer,TOKENS_MAX_LENGTH)
        test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                        batch_size=TRAIN_BATCH_SIZE)

        
        output_scores = predict_fn(test_dataloader,self.model,self.device)
        
        return output_scores,np.argmax(output_scores,axis=1)

if __name__ == '__main__':
    import pandas as pd

    df = pd.read_csv('data/sample.csv')
    mappings = {'O':0, 'IE':1, 'IS':2}
    df['label'].replace({'O':0, 'IE':1, 'IS':2},inplace=True)
    # classifier=Classifier(df,embeddings_model_type= 'sentence_transformer',vectorizer_path ='sentence-transformers/stsb-roberta-large' )
    # classifier.train_svm()

    classifier = BertClassifier(df)
    classifier.train()
    output_scores,output = classifier.predict(df) 
    print(output)