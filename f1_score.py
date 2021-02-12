import collections
import sys
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np


# def f1_score(tags, predicted):
#     tags=set(tags)
#     predicted=set(predicted)

#     tp=len(tags.intersection(predicted))
#     fp=len(predicted.difference(tags))
#     fn=len(tags.difference(predicted))

#     if tp>0:
#         precision=float(tp)/(tp+fp)
#         recall=float(tp)/(tp+fn)


#         return 2*((precision*recall)/(precision+recall))
#     else:
#         return 0

# def confusion_matrix(y_true, y_pred):
#     r = max(y_true)
#     matrix = [[0 for _ in range(r)] for _ in range(r)]
#     for i in range(len(y_true)):
#         matrix[y_true[i]][y_pred[i]] += 1
        
#     return matrix

def get_preds(file_name):
    preds = collections.defaultdict(dict)

    with open(file_name) as file:
        for line in file:
            key, emotion, prob = line.split('\t')
            preds[key][emotion] = prob
        
    emotion_order = {2: 'anger', 6: 'disgust', 4: 'fear', 0: 'happiness', 1: 'no_emotion', 3: 'sadness', 5: 'surprise'}
    prediction_matrix = []

    for key in sorted(list(preds.keys())):
        row = [0 for _ in range(7)]
        for i in range(7):
            row[i] = float(preds[key][emotion_order[i]])
        prediction_matrix.append(row)
    
    prediction_matrix = np.argmax(np.array(prediction_matrix), axis = 1)
    return prediction_matrix
    
preds = get_preds(sys.argv[1])
labels = get_preds(sys.argv[2])

print("F1 score =", f1_score(labels, preds, average='micro', labels=[0,2,3,4,5,6]))
print("Confusion Matrix")
print(confusion_matrix(labels, preds))