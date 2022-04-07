
from operator import mod
from data_reader import csv_reader
from train import Classifier
from evaluator import evaluator


if __name__ == "__main__":
    dataset = csv_reader("data/sample.csv")

    classifier = Classifier(dataframe=dataset,embeddings_model_type='glove',vectorizer_path = None)
    classifier.train_predict()
    eval = evaluator(classifier)
    print (eval.precision())
    print (eval.recall())
    print (eval.accuracy())

    test_dataset = csv_reader("data/addiction.csv")
    classifier = Classifier(dataframe=dataset,embeddings_model_type='glove',vectorizer_path = None)
    pred_list,test_list = classifier.predict(model_path='models/svm.pkl')
    print(pred_list)





