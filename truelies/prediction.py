import pandas as pd
import numpy as np
import random
import pickle
from tkinter import *
from sklearn import model_selection
from sklearn.naive_bayes import MultinomialNB
# Machine Learning Packages
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


root = Tk()

root.title("TRUELIES")
 
root.geometry('300x150')
def ad():
	window = Toplevel(root) 
	lbl = Label(window, text="Please enter the Advertisement you want to verify")
	lbl1 = Label(window, text="") 
	lbl.grid(column=0, row=0)
	lbl1.grid(column=0, row=5 ) 
	txt = Entry(window,width=100)
	lbl2 = Label(window, text="") 
	lbl2.grid(column=0, row=6) 
	txt.grid(column=0, row=1)
	def reset():
	    txt.delete(first=0,last=100)
	    lbl1.configure(text="")
	    lbl2.configure(text="")
	 
	def clicked():
	    load_model = pickle.load(open('final_model.sav', 'rb'))
	    load2=pickle.load(open('finalized_model.sav', 'rb'))
	    var=txt.get()
	    
	    prediction = load_model.predict([var])
	    prob = load_model.predict_proba([var]) 
	    res = "The given statement is ",prediction[0] 
	    lbl1.configure(text= res)
	    res1="The truth probability score is ",prob[0][1]
	    lbl2.configure(text= res1)
	btn = Button(window, text="VERIFY", command=clicked)
	btn1= Button(window, text="RESET",  command=reset)
	btn1.grid(column=0,row=4)
	btn.grid(column=0, row=3)
def link():
	window = Toplevel(root) 
	lbl = Label(window, text="Please enter the Link you want to verify")
	lbl1 = Label(window, text="") 
	lbl.grid(column=0, row=0)
	lbl1.grid(column=0, row=5 ) 
	txt = Entry(window,width=100)
	lbl2 = Label(window, text="") 
	lbl2.grid(column=0, row=6) 
	txt.grid(column=0, row=1)
	def reset():
	    txt.delete(first=0,last=100)
	    lbl1.configure(text="")
	    lbl2.configure(text="")
	 
	def clicked():
	    load_model = pickle.load(open('finalized_model.sav', 'rb'))
	    
	    var=txt.get()
	    var= vectorizer.transform([var])
	    prediction = load_model.predict(var)
	    prob = load_model.predict_proba(var) 
	    res = "The given statement is ",prediction[0] 
	    lbl1.configure(text= res)
	    res1='The truth probability score is ',prob[0][1]
	    lbl2.configure(text= res1)
	btn = Button(window, text="VERIFY", command=clicked)
	btn1= Button(window, text="RESET",  command=reset)
	btn1.grid(column=0,row=4)
	btn.grid(column=0, row=3)
btn = Button(root, text="ad", command=ad)
btn.grid(column=3, row=3)
btn.place(relx=0.5, rely=0.25, anchor=CENTER)
btn1 = Button(root, text="link", command=link)
btn1.grid(column=3, row=4)
btn1.place(relx=0.5, rely=.5, anchor=CENTER)	
root.mainloop()
