#!/usr/bin/env python
# codin
# # In[1]:
#
#
# import streamlit as st
# import pickle
# import numpy as np
# import pandas as pdg: utf-8

# In[1]:


import streamlit as st
import pickle
import numpy as np
import pandas as pd
import string


# In[2]:


model = pickle.load(open("svcGod.pkl","rb"))
vect = pickle.load(open("vectorGod.pkl","rb"))

# In[3]:
# text=["Oh, how my little grandson loves this app. He."]
# text=vect.transform(text)
# text=text.toarray()
# pred=model.predict(text)
# pred=int(pred)
# print(type(pred))

def predict_helpfulness(text):

    input=[text]
    text = vect.transform(input)
    text=text.toarray()
    pred=model.predict(text)
    pred=int(pred)
    if pred == 0:
        #print("Non-Helpful review text has been entered")
        return "Not Helpful"
    elif pred == 1:
        #print("Helpful review text has been entered")
        return "Helpful"
    else:
        #print("Unecessary review text entered")
        return "Not entered properly"



# In[4]:
def main():
    st.title("Amazon Review Text Helpfulness Prediction ")
    text = st.text_input("Text")
    # df1['reviewText'] = df1['reviewText'].str.lower()
    # df1['reviewText']=df1['reviewText'].apply(lambda x: "".join([i for i in x if i not in string.punctuation]))
    # df1['reviewText'].head(4)
    # text = st.text_input("Text")


    if st.button("Predict"):
      output=predict_helpfulness(text)
      st.success("The Given Review Text is {}".format(output))
      
      
      
      


# In[5]:


if __name__=="__main__":
    main()

#
# # In[ ]:
#
#
#
#
#
# # In[ ]:
#
#
#
#
#
# # In[ ]:
#
#
#
#
# streamlit run SentimentAnalysis_Reditt_with_deployment.py
# streamlit run SentimentAnalysis_Reditt_with_deployment.py
# streamlit run SentimentAnalysis_Reditt_with_deployment.py
# import nltk
#
# nltk.download('vader_lexicon')
# from sklearn.svm import SVC
# from textblob import Word
# from sklearn.feature_extraction.text import TfidfVectorizer
# # creating a Sentiment Intensity Analyzer (SIA) object
#
# import streamlit as st
# # https://docs.streamlit.io/en/stable/
#
# import pickle
# import nltk
# import string
#
# nltk.download('stopwords')
# stop = stopwords.words('english')
# from nltk.corpus import stopwords
#
# model = pickle.load(open("svclassifier.pkl" , "rb"))
# vectorizer = pickle.load(open("vectorizer.pkl" , "rb"))
#
#
# # def sia(text):
# #     input = text
# #     predict = model.polarity_scores(input)
# #     return predict
# def clean( text ):
#     nt1 = []
#     new_text = []
#     # for i in text:
#     # 	if i in string.punctuation:
#     # 		continue
#     # 	nt1.append(i)
#     # text=" ".join(nt1)
#     for i in text.split():
#         if i in stop: continue
#         i = Word(i).lemmatize()
#         new_text.append(i.lower())
#     return [" ".join(new_text)]
#
#
# def main():
#     st.title("SENTIMENT ANALYSER ON REDDIT NEWS")
#     text = st.text_input("Text")
#     text = clean(text)
#
#     check = vectorizer.transform(text)
#     if st.button("Predict"):
#         output = model.predict(check)
#         st.success(output)
#
#
# if __name__ == "__main__":
#     main()