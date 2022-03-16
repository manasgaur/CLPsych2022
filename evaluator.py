class evaluator(object):
    '''will calculate precision
       , accuracy, recall, and
       f1 measure
    '''

    def __init__(self,model = None):

        #three classes 
        self.model = model
        self.unique_classes = set(list(self.model.y_test))
        self.c1_n = sum([item for item in list(self.model.y_test) if item == unique_classes[0]])
        self.c2_n = sum([item for item in list(self.model.y_test) if item == unique_classes[1]])
        self.c3_n = sum([item for item in list(self.model.y_test) if item == unique_classes[2]])
        
    def calculate_accuracy(self):
        '''calculates TP+TN/(Total)
           averaged across all three unique classes
        '''

        unique_classes = self.unique_classes
        
        #class 1 accuracy
        y_pred = list(self.model.y_pred)
        y_pred = [0 if item != unique_classes[0] else 1 for item in y_pred]
        y_test = list(self.model.y_test)
        y_test = [0 if item != unique_classes[0] else 1 for item in y_pred]
        n = len(y_test)

        c1_accuracy = sum([y_pred[i]!=y_test[i] for i in range(n)])/float(n)

        #class 2 accuracy
        y_pred = list(self.model.y_pred)
        y_pred = [0 if item != unique_classes[1] else 1 for item in y_pred]
        y_test = list(self.model.y_test)
        y_test = [0 if item != unique_classes[1] else 1 for item in y_pred]
        n = len(y_test)

        c2_accuracy = sum([y_pred[i]!=y_test[i] for i in range(n)])/float(n)

        #class 3 accuracy
        y_pred = list(self.model.y_pred)
        y_pred = [0 if item != unique_classes[2] else 1 for item in y_pred]
        y_test = list(self.model.y_test)
        y_test = [0 if item != unique_classes[2] else 1 for item in y_pred]
        n = len(y_test)

        c3_accuracy = sum([y_pred[i]!=y_test[i] for i in range(n)])/float(n)

        return (self.c1_n*c1+self.c2_n*c2+self.c3_n*c3)/float(n)
        

    def precision(self):
        '''calculate TP/TP+FP
        '''

        unique_classes = self.unique_classes

        #class 1 precision
        y_pred = list(self.model.y_pred)
        y_pred = [0 if item != unique_classes[0] else 1 for item in y_pred]
        y_test = list(self.model.y_test)
        y_test = [0 if item != unique_classes[0] else 1 for item in y_pred]
        n = len(y_test)

        tp = sum([y_pred[i] == 1 and y_test[i] == 1 for i in range(n)])
        fp = sum([y_pred[i] == 1 and y_test[i] == 0 for i in range(n)])

        c1_p = tp/float(tp+fp)

        #class 2 precision
        y_pred = list(self.model.y_pred)
        y_pred = [0 if item != unique_classes[1] else 1 for item in y_pred]
        y_test = list(self.model.y_test)
        y_test = [0 if item != unique_classes[1] else 1 for item in y_pred]
        n = len(y_test)

        tp = sum([y_pred[i] == 1 and y_test[i] == 1 for i in range(n)])
        fp = sum([y_pred[i] == 1 and y_test[i] == 0 for i in range(n)])

        c2_p = tp/float(tp+fp)

        #class 3 precision
        y_pred = list(self.model.y_pred)
        y_pred = [0 if item != unique_classes[2] else 1 for item in y_pred]
        y_test = list(self.model.y_test)
        y_test = [0 if item != unique_classes[2] else 1 for item in y_pred]
        n = len(y_test)

        tp = sum([y_pred[i] == 1 and y_test[i] == 1 for i in range(n)])
        fp = sum([y_pred[i] == 1 and y_test[i] == 0 for i in range(n)])

        c3_p = tp/float(tp+fp)

        return (self.c1_n*c1_p+self.c2_n*c2_p+self.c3_n*c3_p)/float(n)
        

                                                                    
    def recall(self):
        '''calculates TP/TP+FN
        '''

        unique_classes = self.unique_classes

        #class 1 precision
        y_pred = list(self.model.y_pred)
        y_pred = [0 if item != unique_classes[0] else 1 for item in y_pred]
        y_test = list(self.model.y_test)
        y_test = [0 if item != unique_classes[0] else 1 for item in y_pred]
        n = len(y_test)

        tp = sum([y_pred[i] == 1 and y_test[i] == 1 for i in range(n)])
        fn = sum([y_pred[i] == 0 and y_test[i] == 1 for i in range(n)])

        c1_r = tp/float(tp+fp)

        #class 2 precision
        y_pred = list(self.model.y_pred)
        y_pred = [0 if item != unique_classes[1] else 1 for item in y_pred]
        y_test = list(self.model.y_test)
        y_test = [0 if item != unique_classes[1] else 1 for item in y_pred]
        n = len(y_test)

        tp = sum([y_pred[i] == 1 and y_test[i] == 1 for i in range(n)])
        fp = sum([y_pred[i] == 0 and y_test[i] == 1 for i in range(n)])

        c2_r = tp/float(tp+fp)

        #class 3 precision
        y_pred = list(self.model.y_pred)
        y_pred = [0 if item != unique_classes[2] else 1 for item in y_pred]
        y_test = list(self.model.y_test)
        y_test = [0 if item != unique_classes[2] else 1 for item in y_pred]
        n = len(y_test)

        tp = sum([y_pred[i] == 1 and y_test[i] == 1 for i in range(n)])
        fp = sum([y_pred[i] == 0 and y_test[i] == 1 for i in range(n)])

        c3_r = tp/float(tp+fp)

        return (self.c1_n*c1_r+self.c2_n*c2_r+self.c3_n*c3_r)/float(n)
        

def main():

    #import statements
    import pickle
    model = None
    with open('models/svm.pkl','rb') as f:
        model = pickle.load(f)
    eval = evaluator(model)
    print (eval.precision)
    print (eval.recall)
    print (eval.accuracy)
    print (eval.f1_score)

if __name__ == '__main__':
    main()
