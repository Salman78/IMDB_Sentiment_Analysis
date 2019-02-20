
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
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import cross_val_score




def construct_file_list(directory, is_positive):
	file_list_dict = {}
	for text_file in sorted(os.listdir(directory)):
		os.chdir(directory)
		if text_file.endswith(".txt"):
	    	#print(os.getcwd())
			if(is_positive==True):
				file_list_dict[text_file] = "1"
			else:
				file_list_dict[text_file] = "0"
	return file_list_dict
	




def construct_feature_matrix_TfidfVectorizer(pos_file_list_df, neg_file_list_df, maximum_features):
	
	#POSITIVE 
	cv = TfidfVectorizer(input='filename', stop_words='english', max_df=0.75, min_df=5, max_features=maximum_features)
	os.chdir('/Users/Tausal21/Desktop/comp_551/mini_peoject_02/train/pos')
	pos_file_list = pos_file_list_df.FILES.tolist()
	print("pos file list: ", pos_file_list)
	pos_target_y = pd.DataFrame(pos_file_list_df.TARGET_Y.tolist(), columns=['TARGET_Y'])
	print("pos target y: ", pos_target_y)
	

	vec = cv.fit(pos_file_list)

	bag_of_words = vec.transform(pos_file_list)
	pos_feature_matrix_withoutTarget = pd.DataFrame(bag_of_words.toarray(), columns=vec.get_feature_names())

	pos_feature_matrix = pd.concat([pos_feature_matrix_withoutTarget, pos_target_y], axis=1)


	#NEGATIVE
	os.chdir('/Users/Tausal21/Desktop/comp_551/mini_peoject_02/train/neg')
	neg_file_list = neg_file_list_df.FILES.tolist()
	neg_target_y = pd.DataFrame(neg_file_list_df.TARGET_Y.tolist(), columns=['TARGET_Y'])

	vec = cv.fit(neg_file_list)
 
	bag_of_words = vec.transform(neg_file_list)
	neg_feature_matrix_withoutTarget = pd.DataFrame(bag_of_words.toarray(), columns=vec.get_feature_names())

	neg_feature_matrix = pd.concat([neg_feature_matrix_withoutTarget, neg_target_y], axis=1)	

	full_feature_matrix_list = [pos_feature_matrix, neg_feature_matrix]
	full_feature_matrix = pd.concat(full_feature_matrix_list, sort=False).fillna(0)

	return full_feature_matrix


		



def kFold_LogisticRegression(feature_matrix_input, no_of_folds):
	model = LogisticRegression()
	scores = cross_val_score(model, feature_matrix_input.loc[:, feature_matrix_input.columns != 'TARGET_Y'], feature_matrix_input.TARGET_Y, cv=no_of_folds)
	return scores



def split_pos_neg(full_file_list_df):
	pos_file_list_df = full_file_list_df[full_file_list_df['TARGET_Y'] == '1']
	neg_file_list_df = full_file_list_df[full_file_list_df['TARGET_Y'] == '0']

	return [pos_file_list_df, neg_file_list_df]



def Logistic_reg(train_X, train_Y, test_X, test_Y):
	print("train_X shape: ", train_X.shape)
	print("train_Y shape: ", train_Y.shape)
	print("test_X shape ", test_X.shape)
	print("test_Y shape: ", test_Y.shape)
	model = LogisticRegression()
	model.fit(train_X, train_Y)
	model.predict(test_X)
	score = model.score(test_X, test_Y)

	return score




def cross_validation_logisticRegression(full_file_list_df, KFold, maximum_features):
	prediction_score = []
	for i in range(KFold):
		training_set, test_set = train_test_split(full_file_list_df, test_size=0.2, random_state=40)
		print("training set: ", training_set)
		print("test set: ", test_set)
		[train_pos_file_list_df, train_neg_file_list_df] = split_pos_neg(training_set)
		train_full_feature_matrix = construct_feature_matrix_TfidfVectorizer(train_pos_file_list_df, train_neg_file_list_df, maximum_features) #Feature selection is now separately done on training set

		[test_pos_file_list_df, test_neg_file_list_df] = split_pos_neg(test_set)
		test_full_feature_matrix = construct_feature_matrix_TfidfVectorizer(test_pos_file_list_df, test_neg_file_list_df, 4845)  ##Feature selection is now separately done on test set

		prediction_score.append(Logistic_reg(train_full_feature_matrix.loc[:, train_full_feature_matrix.columns != 'TARGET_Y'], train_full_feature_matrix.TARGET_Y, test_full_feature_matrix.loc[:, test_full_feature_matrix.columns != 'TARGET_Y'], test_full_feature_matrix.TARGET_Y))

	return prediction_score







pos_dir = '/Users/Tausal21/Desktop/comp_551/mini_peoject_02/train/pos'
neg_dir = '/Users/Tausal21/Desktop/comp_551/mini_peoject_02/train/neg'
test_dir = '/Users/Tausal21/Desktop/comp_551/mini_peoject_02/test'

pos_file_list_dict = construct_file_list(pos_dir, True)
neg_file_list_dict = construct_file_list(neg_dir, False)


full_file_list_dict = dict(pos_file_list_dict)
full_file_list_dict.update(neg_file_list_dict)

#full_file_list_df = pd.DataFrame.from_dict(full_file_list_dict, orient='index', columns=['TARGET_Y'])
full_file_list_df = pd.DataFrame(list(full_file_list_dict.items()), columns=['FILES', 'TARGET_Y'])
print(full_file_list_df)

prediction_score = cross_validation_logisticRegression(full_file_list_df, 1, 5000)
print(prediction_score)





