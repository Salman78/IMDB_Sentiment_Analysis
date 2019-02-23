import csv
import os
import math
import nltk
import numpy as np
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB

from sklearn import tree
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
import string

#Method that loads the text file passed to it, processes the text by removing stopwords and punctuation, and putting them in a list

def build_features(path,file):
    os.chdir(path)
    file = open(file,"r",encoding='utf-8')
    text = []
    stop_words = set(stopwords.words('english')) 
    for line in file.readlines():
        word_tokens = line.translate(translator)
        word_tokens = word_tokens.split()
        filtered_line = [word for word in word_tokens if word not in stop_words]
        filtered_line = " ".join(filtered_line)
        text.append(filtered_line)
    return text

#Method to create prediction for the test text and build the CSV file to hold the predictions

def predict(train_text,test_set,target_train_matrix):
    with open('predictions_tfidf.csv', mode='w', newline = '') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        train_matrix = vectorizer.fit_transform(train_text)
        test_matrix = vectorizer.transform(test_set)
        model = LogisticRegression()
        model.fit(train_matrix,target_train_matrix)
        prediction = model.predict(test_matrix)
        writer.writerow(['Id', 'Category'])
        to_write = []
        for i in range(0,len(prediction)):
            to_write.append([i,prediction[i]])
        writer.writerows(to_write)

# K Fold validation method for comparing model accuracy

def kfold2(kf,corpus,target):
    target = np.array(target)
    for train_index, test_index in kf.split(corpus,target):
        X_train = [corpus[i] for i in train_index]
        X_test = [corpus[i] for i in test_index]
        y_train, y_test = target[train_index], target[test_index]
        train_corpus_tf_idf = vectorizer.fit_transform(X_train) 
        test_corpus_tf_idf = vectorizer.transform(X_test)

        model1 = DecisionTreeClassifier()
        model2 = LinearSVC()    
        model1.fit(train_corpus_tf_idf,y_train)
        model2.fit(train_corpus_tf_idf,y_train)
        result1 = model1.predict(test_corpus_tf_idf)
        result2 = model2.predict(test_corpus_tf_idf)
        
        print(" DecisionTreeClassifier score using built-in score function: ", model1.score(X_test,y_test))
        print(" SVM score using built-in score function: ", model2.score(X_test,y_test))

        
        return accuracy_score(y_test,result2)

stop_words = set(stopwords.words('english'))
punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
translator = str.maketrans('', '', string.punctuation)
neg_path = '/Users/Tausal21/Desktop/comp_551/mini_peoject_02/train/neg'

# The following lines load the text values from files that have been compressed using the text_compress py file
training_text_neg = build_features(neg_path,"neg_text.txt")

print("training_Text_neg: ", training_text_neg)

print("neg_done")
pos_path = '/Users/Tausal21/Desktop/comp_551/mini_peoject_02/train/pos'

training_text_pos = build_features(pos_path,"pos_text.txt")

print("training_Text_pos: ", training_text_pos)

print("pos_done")
test_path = '/Users/Tausal21/Desktop/comp_551/mini_peoject_02/test'
test_text = build_features(test_path,"test_text.txt")
os.chdir(test_path)
file = open("test_index.txt","r",encoding='utf-8')
# Loads the file index of the test files for later use 
test_index = []
for line in file.readlines():
    test_index.append(int(line))
train_text = training_text_neg+training_text_pos

print("train_text: ", train_text)

print("all done")
target_train_matrix = []
target_validation_matrix = []

# Builds the target matrix for the train values
for i in range(0,12500):
    target_train_matrix.append(0)
for i in range(0,12500):
    target_train_matrix.append(1)
kf = KFold(n_splits = 10)

os.chdir('/Users/Tausal21/Desktop/comp_551/mini_peoject_02')
#Builds the tfidf vectorizer
vectorizer = TfidfVectorizer(min_df=10, max_df = 0.8, ngram_range = (1,2), sublinear_tf=True, use_idf=True,stop_words='english')

accuracy = 0

# Execute K fold validation
accuracy += kfold2(kf,train_text,target_train_matrix)
n_splits = 10


print("AVERAGE: ",accuracy/n_splits)
#Sort the test set in the correct file order
final_test = []
for i in range(0,len(test_index)):
    final_test.append(0)
for i in range(0,len(test_index)):
    final_test[test_index[i]] = test_text[i]

#predict(train_text,final_test,target_train_matrix)
