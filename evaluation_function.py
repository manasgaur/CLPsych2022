import numpy as np

'''
This script contains three sets of evaluation functions:
    (a) post-level eval scores that are calculated by merging the actual/predicted labels across all timelines:
            > precision = get_precision_score(actual, predicted, LABEL)
            > recall = get_recall_score(actual, predicted, LABEL)
            > f1_score = get_f1_score(actual, predicted, LABEL)
    
    (b) timeline-level, window-based precision and recall scores (run on each timeline independently!):
            > precision_w, recall_w = get_timeline_level_precision_recall(actual, predicted, WINDOW, LABEL)
    
    (c) timeline-level, coverage-based precision/recall scores (run on each timeline independently!):
            > cov_recall = get_coverage_recall(actual, predicted, LABEL)
            > cor_precision = get_coverage_precision(actual, predicted, LABEL)
'''


'''(a) The post-level evaluation scripts'''
def get_precision_score(actual, predicted, cls='IS'):
    """
    Returns the overall (across all timelines) precision score for a particular label.
    
    Parameters
    ----------
    actual: list
        list of actual labels in a dev/test set (one per post)
    predicted: list
        list of predicted labels in a dev/test set (one per post)
    cls:
        the label we are after (0/IS/IE)

    Returns
    -------
    prec: float
        the final precision score for the specified label
    """
    assert len(actual)==len(predicted)
    actual, predicted = list(actual), list(predicted)
    if predicted.count(cls)==0:
        print('Made no predictions for label', cls, '(precision undefined - returning nan).')
        return np.nan
    ac_idx = set(np.where(np.array(actual)==cls)[0])
    pr_idx = set(np.where(np.array(predicted)==cls)[0])
    prec = len(ac_idx.intersection(pr_idx))/len(pr_idx)
    return prec


def get_recall_score(actual, predicted, cls='IS'):
    """
    Returns the overall (across all timelines) recall score for a particular label.
    
    Parameters
    ----------
    actual: list
        list of actual labels in a dev/test set (one per post)
    predicted: list
        list of predicted labels in a dev/test set (one per post)
    cls:
        the label we are after (0/IS/IE)

    Returns
    -------
    rec: float
        the final recall score for the specified label
    """
    actual, predicted = list(actual), list(predicted)
    assert len(actual)==len(predicted)
    if actual.count(cls)==0:
        print('Have no examples for label', cls, '(recall undefined - returning nan).')
        return np.nan
    ac_idx = set(np.where(np.array(actual)==cls)[0])
    pr_idx = set(np.where(np.array(predicted)==cls)[0])
    rec = len(ac_idx.intersection(pr_idx))/len(ac_idx)
    return rec
    

def get_f1_score(actual, predicted, cls='IS'):
    """
    Returns the overall (across all timelines) F1 score for a particular label.
    
    Parameters
    ----------
    actual: list
        list of actual labels in a dev/test set (one per post)
    predicted: list
        list of predicted labels in a dev/test set (one per post)
    cls:
        the label we are after (0/IS/IE)

    Returns
    -------
    f1_sco: float
        the final f1 score for the specified label
    """
    recall = get_recall_score(actual, predicted, cls)
    precision = get_precision_score(actual, predicted, cls)
    f1_sco = 2.0*(recall*precision)/(recall+precision)
    return f1_sco



'''(b) The window-based, timeline-level Precision and Recall script'''
def get_timeline_level_precision_recall(actual, predicted, window=1, cls='IS'):
    """
        Given the lists of (ORDERED!) predicted and actual labels of a SINGLE timeline, 
        the label to calculate the  metrics for and the window to use (allowing 
        +-window predictions to be considered as accurate), it returns:
            (a) the precision using that window for the specified label
            (b) the recall -//-
    
    Parameters
    ----------
    actual: list
        list of actual labels (ORDERED) in a single timeline (one per post)
    predicted: list
        list of actual labels (ORDERED) in a single timeline (one per post)
    window: int
        the window size to consider (you can play around with increasing/decreasing)
    cls:
        the label we are after (0/IS/IE)

    Returns
    -------
    precision: float
        the final window-based precision score for the specified label in this timeline
    recall: float
        the final window-based recall score for the specified label in this timeline
    """
    assert len(actual)==len(predicted)
    if (len(actual)>125) or (len(actual)<10):
        print('This function should be run at the timeline-level (i.e., not by merging all actual/predicitons together)!')

    # Find the indices of the specified predicted and actual label
    pr_idx = np.where(np.array(predicted)==cls)[0]
    ac_idx = np.where(np.array(actual)==cls)[0]

    if len(ac_idx)==0: # cannot divide by zero (Recall is undefined)
        recall, precision = np.nan, np.nan
        if len(pr_idx)>0:
            precision = 0.0
    elif len(pr_idx)==0: # cannot divide by zero (Precision is undefined, but Recall is 0)
        precision = np.nan
        recall = 0.0
    else:
        already_used = []
        for l in ac_idx: 
            for p in pr_idx: 
                if (np.abs(l-p)<=window) & (p not in already_used): 
                    already_used.append(p) 
                    break 
        precision = len(set(already_used))/len(pr_idx)
        recall = len(set(already_used))/len(ac_idx)
    return precision, recall


'''(c) The coverage scripts'''
def get_coverage_recall(actual, predicted, cls='IS'):
    """
        Given the lists of (ORDERED!) predicted and actual labels of a SINGLE timeline, 
        the label to calculate the  metrics for and the window to use (allowing 
        +-window predictions to be considered as accurate), it returns the recall-oriented
        coverage for that particular timeline
    
    Parameters
    ----------
    actual: list
        list of actual labels (ORDERED) in a single timeline (one per post)
    predicted: list
        list of actual labels (ORDERED) in a single timeline (one per post)
    cls:
        the label we are after (0/IS/IE)

    Returns
    -------
    coverage_recall: float
        the final coverage-based recall score for the specified label in this timeline
    """
    assert len(actual)==len(predicted)
    if (len(actual)>125) or (len(actual)<10):
        print('This function should be run at the timeline-level (i.e., not by merging all actual/predicitons together)!')

    preds_regions, _ = extract_regions(predicted, cls)
    actual_regions, _ = extract_regions(actual, cls)

    total_sum, denom = 0.0, 0.0 # timeline basis
    coverage_recall = np.nan
    if len(actual_regions)>0:
        for region in actual_regions: # For each actual region within the timeline
            ac = set(region)
            Orrs = [] #calculated per region
            max_cov_for_region = 0.0 #calculated per region

            #Find the maximum ORR for this actual region:
            if len(preds_regions)>0: 
                for predicted_region in preds_regions: 
                    pr = set(predicted_region)
                    Orrs.append(len(ac.intersection(pr))/len(ac.union(pr))) # Intersection over Union
                max_cov_for_region = np.max(Orrs)
            
            #Now multiply it by the length of the region
            total_sum = total_sum + (len(ac)*max_cov_for_region)
            denom += len(ac)            
        coverage_recall = total_sum/denom
    return coverage_recall


def get_coverage_precision(actual, predicted, cls='IS'):
    """
        Given the lists of (ORDERED!) predicted and actual labels of a SINGLE timeline, 
        the label to calculate the  metrics for and the window to use (allowing 
        +-window predictions to be considered as accurate), it returns the precision-oriented
        coverage for that particular timeline
    
    Parameters
    ----------
    actual: list
        list of actual labels (ORDERED) in a single timeline (one per post)
    predicted: list
        list of actual labels (ORDERED) in a single timeline (one per post)
    cls:
        the label we are after (0/IS/IE)

    Returns
    -------
    coverage_precision: float
        the final coverage-based precision score for the specified label in this timeline
    """
    assert len(actual)==len(predicted)
    if (len(actual)>125) or (len(actual)<10):
        print('This function should be run at the timeline-level (i.e., not by merging all actual/predicitons together)!')

    actual_regions, _ = extract_regions(actual, cls)
    preds_regions, _ = extract_regions(predicted, cls)

    total_sum, denom = 0.0, 0.0 # big sum and 1/N, respectively
    coverage_precision = np.nan
    if len(preds_regions)>0:
        for region in preds_regions: # For each predicted region within the timeline
            ac = set(region)
            Orrs = []
            max_cov_for_region = 0.0

            #Find the maximum ORR for this predicted region:
            if len(actual_regions)>0: 
                for predicted_region in actual_regions: 
                    pr = set(predicted_region)
                    Orrs.append(len(ac.intersection(pr))/len(ac.union(pr))) # Intersection over Union
                max_cov_for_region = np.max(Orrs)
            
            total_sum = total_sum + (len(ac)*max_cov_for_region)
            denom += len(ac)            
        coverage_precision = total_sum/denom
    return coverage_precision


'''Helper functions for the coverage-based metrics'''
def extract_regions(vals, cls):
    #Convert labels to boolean based on the class we are looking for:
    vals = np.array(vals)
    vals_boolean = np.zeros(len(vals))
    vals_boolean[vals==cls] = 1

    #Find the indices of the positive cases:
    indices = np.where(vals_boolean==1)[0]
    neg_indices = np.where(vals_boolean==0)[0]
    
    actual_regions = get_regions(indices)
    actual_regions_neg = get_regions(neg_indices)
    return actual_regions, actual_regions_neg


def get_regions(indices):
    actual_regions = []         
    if len(indices)>0:
        current_set = [indices[0]]
        for i in range(1, len(indices)):
            if indices[i]-indices[i-1]==1: #if they are consecutive
                current_set.append(indices[i])
            else:
                actual_regions.append([v for v in current_set])
                current_set = [indices[i]]
        actual_regions.append(current_set)
    return actual_regions