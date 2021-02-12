#	Only for dailydialog dataset
#	Run Info
#	run this as python data_preprocess.py <location of folder with the three json files>
#
#	The three json files are
#	train.json, valid.json and test.json

import json
import pandas as pd
import numpy as np
import sys

def read_file(path):
    data = []
    with open(path) as file:
        for line in file:
            data.append(json.loads(line))
    return data

def extract_fields(data, suff):
    flattened_data = []
    conv_list = []
    for i,conv in enumerate(data):
        dial = []
        for j,act in enumerate(conv['dialogue']):
            flattened_data.append([suff+str(i)+'_'+str(j), act['text'], act['emotion'], act['act']])
            dial.append([suff+str(i)+'_'+str(j), act['text'], act['emotion'], act['act']])
        conv_list.append(dial)
    df = pd.DataFrame(flattened_data, columns=['UID','text','emotion','act'])
    
    return df, conv_list

def create_better_graph_edges(conversations, cv, tfidf, label_map, clf):
    WhatISaid = [] # Self Explanatory
    WhatYouSaid = [] # Self Explanatory
    UttEmo = [] # Emotion in the existing dialogue eg. (1600, 'no_emotion')
    LocalPred = [] # Predicted emotion of existing dialogue eg. (1600, 'no_emotion', 0.81), (1600, 'happiness', 0.14), ...

    for dial in conversations:
        WIS = []
        WYS = []
        Stat_Emo = []
        Utt_Emo = []
        
        spks = []
        liss = []

        for idx,utt in enumerate(dial):
            
            # Get predictions
            utt_cv = cv.transform([utt[1]])
            utt_tfidf = tfidf.transform(utt_cv)
            
            utt_pred = clf.predict_proba(utt_tfidf)
            for i in range(len(utt_pred[0])):
                Utt_Emo.append([utt[0], label_map[i], utt_pred[0][i]])
                
            for i in range(len(label_map)):
                if label_map[i] == utt[2]:
                    Stat_Emo.append([utt[0], label_map[i], 1.0])
                else:
                    Stat_Emo.append([utt[0], label_map[i], 0.0])
            
            # Context Edges
            if idx % 2 == 0:
                if spks:
                    WIS.append([utt[0], spks[-1]])
                if liss:
                    WYS.append([utt[0], liss[-1]])
            else:
                if liss:
                    WIS.append([utt[0], liss[-1]])
                if spks:
                    WYS.append([utt[0], spks[-1]])
            
            if idx % 2 == 0:
                spks.append(utt[0])
            else:
                liss.append(utt[0])

        WhatISaid.append(WIS)
        WhatYouSaid.append(WYS)
        UttEmo.append(Stat_Emo)
        LocalPred.append(Utt_Emo)
    
    return WhatISaid, WhatYouSaid, UttEmo, LocalPred

def print_file(source, path, mode = 'w'):
    f = open(path, mode)
    
    for dial in source:
        for e in dial:
            f.write('\t'.join(map(str, e)) + '\n')



train_data = read_file(sys.argv[1] + '/train.json')
valid_data = read_file(sys.argv[1] + '/valid.json')
test_data = read_file(sys.argv[1] + '/test.json')

train_df, train_conv = extract_fields(train_data, 'tr_c')
valid_df, valid_conv = extract_fields(valid_data, 'va_c')
test_df, test_conv = extract_fields(test_data, 'te_c')

# Preprocess the train data
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn import preprocessing

CV = CountVectorizer()
train_CV = CV.fit_transform(train_df['text'])
tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(train_CV)

lb = preprocessing.LabelEncoder()
train_labels = lb.fit_transform(train_df['emotion'])
label_map = {}
for i in range(len(lb.classes_)):
    label_map[i] = lb.classes_[i]
print(label_map)


# Build a Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

clf = MultinomialNB(alpha=.01)
clf.fit(train_tfidf, train_labels)

WhatISaid_train, WhatYouSaid_train, UttEmo_train, LocalPred_train = create_better_graph_edges(train_conv, CV, tfidf, label_map, clf)
WhatISaid_valid, WhatYouSaid_valid, UttEmo_valid, LocalPred_valid = create_better_graph_edges(valid_conv, CV, tfidf, label_map, clf)
WhatISaid_test, WhatYouSaid_test, UttEmo_test, LocalPred_test = create_better_graph_edges(test_conv, CV, tfidf, label_map, clf)

# print seperately
print_file(WhatISaid_train, 'WhatISaid_train.txt')
print_file(WhatYouSaid_train, 'WhatYouSaid_train.txt')
print_file(UttEmo_train, 'UttEmo_train.txt')
print_file(LocalPred_train, 'LocalPred_NB_train.txt')

print_file(WhatISaid_valid, 'WhatISaid_valid.txt')
print_file(WhatYouSaid_valid, 'WhatYouSaid_valid.txt')
print_file(UttEmo_valid, 'UttEmo_valid.txt')
print_file(LocalPred_valid, 'LocalPred_NB_valid.txt')

print_file(WhatISaid_test, 'WhatISaid_test.txt')
print_file(WhatYouSaid_test, 'WhatYouSaid_test.txt')
print_file(UttEmo_test, 'UttEmo_test.txt')
print_file(LocalPred_test, 'LocalPred_NB_test.txt')

# print for PSL
print_file(WhatISaid_train, 'WhatISaid.txt')
print_file(WhatYouSaid_train, 'WhatYouSaid.txt')
print_file(UttEmo_train, 'UttEmo.txt')
print_file(LocalPred_train, 'LocalPred_NB.txt')

print_file(WhatISaid_valid, 'WhatISaid.txt', 'a')
print_file(WhatYouSaid_valid, 'WhatYouSaid.txt', 'a')
print_file(UttEmo_valid, 'UttEmo_targets.txt')
print_file(LocalPred_valid, 'LocalPred_NB.txt', 'a')

print_file(WhatISaid_test, 'WhatISaid.txt', 'a')
print_file(WhatYouSaid_test, 'WhatYouSaid.txt', 'a')
print_file(UttEmo_test, 'UttEmo_targets.txt', 'a')
print_file(LocalPred_test, 'LocalPred_NB.txt', 'a')