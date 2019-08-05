import pandas as pd
import numpy as np
import random
import pickle
from tkinter import *
from sklearn import model_selection
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
urls_data = pd.read_csv("urldata.csv")
type(urls_data)  
def makeTokens(f):
    tkns_BySlash = str(f.encode('utf-8')).split('/')	# make tokens after splitting by slash
    total_Tokens = []
    for i in tkns_BySlash:
        tokens = str(i).split('-')	# make tokens after splitting by dash
        tkns_ByDot = []
        for j in range(0,len(tokens)):
            temp_Tokens = str(tokens[j]).split('.')	# make tokens after splitting by dot
            tkns_ByDot = tkns_ByDot + temp_Tokens
        total_Tokens = total_Tokens + tokens + tkns_ByDot
    total_Tokens = list(set(total_Tokens))	#remove redundant tokens
    if 'com' in total_Tokens:
        total_Tokens.remove('com')	#removing .com since it occurs a lot of times and it should not be included in our features
    return total_Tokens
y = urls_data["label"]
url_list = urls_data["url"]
vectorizer = TfidfVectorizer(tokenizer=makeTokens)
#print ( vectorizer )
X = vectorizer.fit_transform(url_list)
#print(vectorizer.get_feature_names())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
nb = MultinomialNB()	
nb.fit(X_train, y_train)
filename = 'finalized_model.sav'
pickle.dump(nb, open(filename, 'wb'))
print(urls_data.label.value_counts())

print("Accuracy ",nb.score(X_test, y_test))

'''
jaba=input("enter")
X_predict = [jaba]

X_predict = vectorizer.transform(X_predict)
print("hi",X_predict)
New_predict = nb.predict(X_predict)
print(New_predict[0]) 
New_prob = nb.predict_proba(X_predict)
print(New_prob[0][1])
'''
def plot_learing_curve(nb,title):
    size = 10
    cv = 3
    
    X = vectorizer.fit_transform(url_list)
    y = urls_data["label"]
    
    pl = nb
    pl.fit(X,y)
    
    train_sizes, train_scores, test_scores = learning_curve(pl, X, y, n_jobs=-1, cv=cv, train_sizes=np.linspace(.1, 1.0, 5), verbose=0)
       
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
     
    plt.figure()
    plt.title(title)
    plt.legend(loc="best")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.gca().invert_yaxis()
    
    # box-like grid
    plt.grid()
    
    # plot the std deviation as a transparent range at each training set size
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
    # plot the average training and test score lines at each training set size
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    
    # sizes the window for readability and displays the plot
    # shows error from 0 to 1.1
    plt.ylim(-.1,1.1)
    plt.show()


#below command will plot learing curves for each of the classifiers
'''

plot_learing_curve(nb_pipeline_ngram,"Naive-bayes Classifier")
by plotting the learning cureve for logistic regression, it can be seen that cross-validation score is stagnating throughout and it 
is unable to learn from data. Also we see that there are high errors that indicates model is simple and we may want to increase the
model complexity.
'''
plot_learing_curve(nb,"Naive-bayes Classifier")
def plotbar():

                label = ['good','bad']
		
                pos = np.arange(2)
                Index = [344821,75643]
                plt.bar(pos, Index, color=['red','blue'],edgecolor='black')
                plt.xticks(pos, label)
                plt.xlabel('LABELS', fontsize=16)
                plt.ylabel('COUNT', fontsize=16)
                plt.title('DATASET LINKS', fontsize=20)
                plt.show()
plotbar()
