"""Train machine learning classifier."""
from typing import Any, List
import pickle
import numpy as np
from model_embeddings import modelEmbeddings
from sklearn import svm
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import classification_report,accuracy_score
from bert_trainer import BERTBaseUncased
from data_reader import BertDataset
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from utils import TRAIN_BATCH_SIZE,EPOCHS
import torch
from bert_trainer import train_fn,eval_fn
import torch
# torch.zeros(1).cuda()

from data_reader import get_train_users

class Classifier:
    def __init__(self,dataframe:pd.DataFrame,embeddings_model_type, vectorizer_path=None, train_vectorizre= True, save_dir = 'models/')-> None:
        """Inititalize classifier."""
        if train_vectorizre:
            self.embeddings_model = modelEmbeddings(embeddings_model_type)
        else:
           self.embeddings_model =  modelEmbeddings(embeddings_model_type,save_path=save_dir)
        self.custom_vectorizer_path = vectorizer_path
        self.saved_models_path = save_dir
        self.create_split(dataframe)
        
    def create_split(self,dataframe):
        """Return train/test embeddings and class"""

        train_df,test_df = train_test_split(dataframe,test_size=0.1,random_state=42, stratify=dataframe.label.values)
        
        self.train_df = train_df
        self.test_df = test_df
        self.x_train = self.embeddings_model(train_df['content'].values)
        self.y_train = train_df['label']
        
        self.x_test = self.embeddings_model(test_df['content'].values)
        self.y_test = test_df['label']

        

    def get_train_test_split(self):
        return self.x_train,self.x_test,self.y_train,self.y_test
    
    def train_predict(self,)->None:
        """Train model.
        
        Parameters
        ----------
        model: Any

        """
        
        X_train, X_test, y_train, y_test = self.get_train_test_split()

        self.svm_model = svm.SVC(kernel='linear', C=3).fit(X_train, y_train)
        self.y_pred = self.svm_model.predict(X_test)
        save_loc = self.save_dir+'svm.pkl'

        print('Accuracy: SVM model = '+str(round(accuracy_score(y_test,self.y_pred)*100,2)))
        print(classification_report(y_test,self.y_pred))

        pickle.dump(self.svm_model,open(save_loc,'wb'))
        print("Model saved at: {}".format(save_loc))

        

class BertClassifier:
    def __init__(self,dataframe:pd.DataFrame,device='cuda',)-> None:
        """"""
        self.dataframe = dataframe
        self.create_split(self.dataframe)

        self.model = BERTBaseUncased()
        self.device = device
        self.model.to(self.device)

    def create_split(self,dataframe):
        """Return train/test embeddings and class"""

        train_df,dev_df = train_test_split(dataframe,test_size=0.1,random_state=42, stratify=dataframe.label.values)
        
        self.train_df = train_df
        self.dev_df = dev_df

        
        train_dataset = BertDataset(
                       text=self.train_df.text.values,
                       target=self.train_df.label.values)
        self.train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=TRAIN_BATCH_SIZE,
                                                        )


        valid_dataset = BertDataset(
                            text= self.dev_df.text.values,
                            target= self.dev_df.label.values)
        self.valid_data_loader = torch.utils.data.DataLoader(valid_dataset,
                                                        batch_size=TRAIN_BATCH_SIZE)

    
    def train(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [{
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],"weight_decay": 0.001,},
            {   "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],"weight_decay": 0.0,}]
        optimizer = AdamW(optimizer_parameters, lr=2e-5)

        num_training_steps = int(len(self.train_df) / TRAIN_BATCH_SIZE * EPOCHS)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,num_training_steps=num_training_steps)

        best_accuracy=0
        
        for ephoch in range(EPOCHS):
            train_fn( self.train_data_loader,self.model,optimizer,self.device,scheduler)
            outputs, targets = eval_fn(self.valid_data_loader,self.model,self.device)
            #Change the accuracy calc below
            outputs = np.array(outputs)>=0.5
            accuracy = accuracy_score(targets, outputs)
            print(f"Accuracy Score = {accuracy} for Epoch = {ephoch} ")
            if accuracy > best_accuracy:
                    torch.save(self.model.state_dict(), "models/bert_classifier.bin")
                    best_accuracy = accuracy

if __name__ == '__main__':
    import pandas as pd

    #
    users = get_train_users()
    dfs = []
    for user in users.keys():
        for i in range(len(users[user]['data'])):
            tdf = users[user]['data'][i]
            tdf['timeline_id'] = users[user]['timelines'][i]
            dfs.append(tdf)
    df = pd.concat(dfs)

    df['content'] = df['content'].fillna(' ')

    df["label"].replace({"0": 1, 0: 1, "IE": 2, "IS": 3}, inplace=True)
   
    classifier = Classifier(df, embedding_type="glove")
    classifier.train_predict()

    classifier = BertClassifier(df)
    classifier.train()

