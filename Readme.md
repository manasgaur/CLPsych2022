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
Select embeddings model type. In this repository, we provide three ways to define numerical embeddings of the textual data. (a) TF-IDF, (b) Sentence Transformer, and (c) GLoVE. You can use either of these incorporated embedding methods or introduce your own by adding another if-else block inside the __call__ function. For instance, BERT embeddings can be used as described in from; https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/ . 
```
from model_embeddings import modelEmbeddings
embeddings_model = modelEmbeddings(model_type = `glove')
embeddings = embeddings_model(documents)
```

Here `model_type` can take following values
  * `glove` : Glove embeddings
  * `tfidf` : tf-idf vectorizer
  * `sentence_transformer` : bert-base-uncased pre-trained embeddings; stsb-roberta-large (https://huggingface.co/sentence-transformers/stsb-roberta-large)
  *  Loction for other complex `model_type`'s that can be used: https://huggingface.co/sentence-transformers/

There is a subtle difference tf-idf and embedding models lies in the engineered features. TF-IDF is like bag of words, discrete, whereas embedding models are continous semantic representation of words or sentences. Best way to select the `model_type` is by computing the similarity between words. Project this similarity into T-SNE or heatmap to analyze which `model_type`'s word similarity scores are sensible, intuitively.
 
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


### Download Embeddings
This repository would store all your pre-trained or fine-tuned embedding models. Also, we suggest storing your trained models here (.pkl (pickle), .npy (numpy), .hd5 are good methods to store trained models.)

There are the two sources from where you can download the GLoVE Embeddings:
* https://nlp.stanford.edu/projects/glove/
* https://github.com/stanfordnlp/GloVe

If you are interested in converting GLoVE to word2vec, a good resource is https://radimrehurek.com/gensim/scripts/glove2word2vec.html

Download Word2Vec Embeddings: https://radimrehurek.com/gensim/models/word2vec.html

One Stop Shops for Embeddings: 
* https://pypi.org/project/embeddings/ 
* http://vectors.nlpl.eu/repository/
* https://developer.syn.co.in/tutorial/bot/oscova/pretrained-vectors.html 

For issues, please email: mgaur@email.sc.edu
