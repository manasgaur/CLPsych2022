
from data_reader import csv_reader
from train import Classifier
from evaluator import evaluator

if __name__ == "__main__":
    dataset = csv_reader("data/sample.csv")
    classifier = Classifier(dataframe=dataset,embeddings_model_type='glove')
    classifier.train_predict()
    eval = evaluator(classifier)
    print (eval.precision())
    print (eval.recall())
    print (eval.accuracy())








