
import os
import pandas as pd
import numpy as np
import nltk
from bs4 import BeautifulSoup 
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import PorterStemmer #not used due to poor performance
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import cross_val_score

def construct_file_list(directory):
	file_list = []
	for text_file in sorted(os.listdir(directory)):
		os.chdir(directory)
		if text_file.endswith(".txt"):
	    	#print(os.getcwd())
			file_list.append(text_file)
	if(directory == '/Users/Tausal21/Desktop/comp_551/mini_peoject_02/test'):
		#file_list.sort(key=lambda f: int(filter(str.isdigit, f)))
		file_list = sorted(file_list , key=lambda x: int(os.path.splitext(x)[0]))
	return file_list

def construct_feature_matrix_countVectorizer(directory, maximum_features, target_column_value, boolean_train):
	file_list = construct_file_list(directory)

	cv = CountVectorizer(input='filename', stop_words='english', max_df=0.75, min_df=5, max_features=maximum_features)
	vec = cv.fit(file_list)

	bag_of_words = vec.transform(file_list)
	feature_matrix_withoutTarget = pd.DataFrame(bag_of_words.toarray(), columns=vec.get_feature_names())

	if(boolean_train == False):
		file_list_df = pd.DataFrame({'col': file_list})
		#print(file_list_df)
		return [feature_matrix_withoutTarget, file_list_df] #This returns the test_set feature matrix
	else:
		if(target_column_value == 1):
			Y_raw = np.ones((12500,), dtype=float)
			target_column_Y = pd.DataFrame(Y_raw, columns=['TARGET_Y']) #appends target column for positive review
		else:
			Y_raw = np.zeros((12500,), dtype=float)
			target_column_Y = pd.DataFrame(Y_raw, columns=['TARGET_Y']) #appends target column for positive review
		
		feature_matrix = pd.concat([feature_matrix_withoutTarget, target_column_Y], axis=1)
		return feature_matrix #returns pos/neg training feature matrix with labels

def kFold_LogisticRegression(feature_matrix_input, no_of_folds):
	model = LogisticRegression()
	scores = cross_val_score(model, feature_matrix_input.loc[:, feature_matrix_input.columns != 'TARGET_Y'], feature_matrix_input.TARGET_Y, cv=no_of_folds)
	return scores

def Logistic_reg(train_X, train_Y, test_X):
	print("train_X shape: ", train_X.shape)
	print("train_Y shape: ", train_Y.shape)
	model = LogisticRegression()
	model.fit(train_X, train_Y)

	prediction_list = model.predict(test_X)
	return prediction_list



pos_dir = '/Users/Tausal21/Desktop/comp_551/mini_peoject_02/train/pos'
neg_dir = '/Users/Tausal21/Desktop/comp_551/mini_peoject_02/train/neg'
test_dir = '/Users/Tausal21/Desktop/comp_551/mini_peoject_02/test'

pos_features = construct_feature_matrix_countVectorizer(pos_dir, 5000, 1, True)
print("Pos_features shape: ", pos_features.shape)
neg_features = construct_feature_matrix_countVectorizer(neg_dir, 5000, 0, True)
print("Neg_features shape: ", neg_features.shape)

feature_matrix_list = [pos_features, neg_features]
feature_matrix_final = pd.concat(feature_matrix_list, sort=False).fillna(0)
print("final_feature shape: ", feature_matrix_final.shape)
no_of_features = len(feature_matrix_final.columns) - 1
print("number of features: ", no_of_features)

[test_set, file_list_df] = construct_feature_matrix_countVectorizer(test_dir, no_of_features, 0, False)
print("test_set shape: ", test_set.shape)
print(file_list_df)

#scores = kFold_LogisticRegression(feature_matrix_final, 10)
prediction_list = Logistic_reg(feature_matrix_final.loc[:, feature_matrix_final.columns != 'TARGET_Y'], feature_matrix_final.TARGET_Y, test_set)
prediction_list_df = pd.DataFrame(prediction_list)


output_df = pd.concat([file_list_df, prediction_list_df], axis=1)
output_df.to_csv('output2.csv')
print(output_df)




