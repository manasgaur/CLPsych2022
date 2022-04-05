**CLPsych 2022 Shared Task Structure**

* Organize the files _(Next 2 weeks)_
* Task : Predict the moments of change in the posts made by the user. Following are some functions needed to create baseline models 
    * data_reader.py : Show take as input certain path to a training dataset containing all the timelines 
    * evaluator.py : Make custom functions for Precision, Recall, F1-Score, and other relevant metrics
    * utils.py : store the results inside utils. 
    * model_building.py : Deep language models specific for the task. Numpy and Torch are acceptable
* Each data point would be an array of:
    * Timeline ID
    * Post ID
    * User ID
    * Date
    * Label : ['IS', 'IE', 'O']
    * text : Post made a user at particular instance of time.
* While assessing the baseline models, we are specifically interested in 'IS' and 'IE' labels.

## Usage
### Loading dataset
You can simply load dataset by inputing file name.
```
dataset = csv_reader("data/sample.csv")
```
### Create embeddings
Select embeddings model type
```
from model_embeddings import modelEmbeddings
embeddings_model = modelEmbeddings(model_type = `glove')
embeddings = embeddings_model(documents)
```
Here `model_type` can take following values
  * `glove` : Glove embeddings
  * `tfidf` : tf-idf vectorizer
  * `sentence_transformer` : bert-base-uncased pre-trained embeddings
 
Loading pre-trained embeddings
```
from model_embeddings import modelEmbeddings
embeddings_model = modelEmbeddings(model_type = `tfidf',load_path='models/tfidf_vectorizer.pkl')
embeddings = embeddings_model(documents)
```

Saving trained model to custom location
```
from model_embeddings import modelEmbeddings
embeddings_model = modelEmbeddings(model_type = `tfidf',save_path='models/tfidf_vectorizer.pkl')
embeddings = embeddings_model(documents)
```

### Training
Train basic SVM classifier and save trained model to custom location
```
dataset = csv_reader("data/sample.csv")
from data_reader import csv_reader
from train import Classifier
dataset = csv_reader("data/sample.csv")
classifier = Classifier(dataframe=dataset,embedding_type='glove',train_vectorizre = False, save_dir='models/')
classifier.train_predict()
```

### Evaluation
Evaluate trained model using evaluator
```
from evaluator import evaluator
eval = evaluator(classifier)
print (eval.precision())
print (eval.recall())
print (eval.accuracy())
```
