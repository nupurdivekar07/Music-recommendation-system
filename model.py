# import pandas as pd
# import pickle
# column_name =["exp","id","time","query","id-name","comments"]
# df=pd.read_csv("D:/Python/sentiment/sentiment.csv",encoding="latin1",names=column_name)
# print(df)
# print(df[["exp","comments"]])
# df['com'] = df['comments']
# print(df["com"])
# ## data preprocessing
# import re
# pr=r'[^\w\s]'
# df["coms"]=df["com"].replace(pr," ",regex=True)
# print(df["coms"])

# import re
# pe=r'http'
# df["coms"]=df["coms"].replace(pe," ",regex=True)
# print(df["coms"])

# import re
# pe1 = "[0-9]"
# df["coms"] =df["coms"].replace(pe1," ",regex=True)
# print(df["coms"])

# import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import stopwords
# stop_words = stopwords.words('english')

# df['coms'] = df['coms'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
# print(df["coms"])

# import nltk
# from nltk.tokenize import sent_tokenize, word_tokenize
# nltk.download('punkt')
# df['coms'] = df.apply(lambda row: nltk.word_tokenize(row['coms']), axis=1)
# print(df['coms'])
# from nltk.stem import PorterStemmer, WordNetLemmatizer
# porter_stemmer = PorterStemmer()
# df['coms'] = df['coms'].apply(lambda x: [porter_stemmer.stem(y) for y in x])
# print(df["coms"])
# df["coms"]=df["coms"].astype(str)
# print(df["coms"].dtype)
# ## model training
# X= df.iloc[:,-1]
# Y= df.iloc[:,0:1]
# from sklearn.model_selection import train_test_split

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# from sklearn.feature_extraction.text import TfidfVectorizer

#  #Initialize TF-IDF Vectorizer
# tfidf_vectorizer = TfidfVectorizer()

# #  Fit and Transform
# tfidf_matrix = tfidf_vectorizer.fit_transform(X_train)
# tfid_X_Test=tfidf_vectorizer.transform(X_test)

#  # Check shape of TF-IDF matrix
# print("TF-IDF Matrix Shape:", tfidf_matrix.shape)

# # tfidf_matrix
# Y
# from sklearn.linear_model import LogisticRegression
# lr=LogisticRegression()
# lr.fit(tfidf_matrix, Y_train)
# predict=lr.predict(tfid_X_Test)
# print(predict)
# from sklearn.metrics import confusion_matrix,accuracy_score
# c_result=confusion_matrix(predict,Y_test)
# acc_score=accuracy_score(predict,Y_test)
# print(c_result)
# print(acc_score)

# ex=["he is bad person"]
# ex_count=tfidf_vectorizer.transform(ex)
# # pre = predict(ex_count)
# # print("Prediction: ",pre)

# predicts=lr.predict(ex_count)
# print(predicts)
# if predicts==[0]:
#     print("it is sad statement")
# else:
#     print("it is positive statement")    

# import pickle

# # Save the trained logistic regression model as a pickle file
# with open('logistic_regression_model.pkl', 'wb') as file:
#     pickle.dump(lr, file)

# # Save the TF-IDF vectorizer as a pickle file
# with open('tfidf_vectorizer.pkl', 'wb') as file:
#     pickle.dump(tfidf_vectorizer, file)







## music





import numpy as np
import pandas as pd
# ab = pd.read_csv("D:\Python\sentiment\songs_5000.csv.xls")

# import os

# # Specify the path to the file
# file_path = "D:/Python/sentiment/song_5000.csv.csv"

# # Rename the file by removing the duplicated extension
# os.rename(file_path, "D:/Python/sentiment/song_5000.csv")

ab = pd.read_csv("D:\Python\sentiment\song_5000.csv")

print(ab)
print("No. of rows and columns: ",ab.shape)  #  Prints the shape (number of rows and columns) of the DataFrame 'ab
print(ab.head(5))
print(ab['song'])
print("(-->)",ab['text'][0]) #Prints the first element of the 'text' column of the DataFrame 'ab'.

#  Converts the text in the 'text' column to lowercase, removes punctuation, 
#and replaces newline characters with spaces.
ab['text'] = ab['text'].str.lower().replace(r'[^\w\s]','').replace(r'\n',' ',regex=True)

print("[-->]",ab['text'][0])
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()  # used to reduce words to its base form 


def tokenization(txt):
    tokens = nltk.word_tokenize(txt)
    stemming = [ps.stem(w) for w in tokens]

    return " ".join(stemming)


# Applies the tokenization function to each element of the 'text' column in the DataFrame 'ab'.
ab['text'] = ab['text'].apply(lambda x: tokenization(x))

print(ab['text'][0])

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tfid = TfidfVectorizer(stop_words='english')
matrix = tfid.fit_transform(ab['text'])

print(matrix.shape)

similarity = cosine_similarity(matrix)

print("-->",similarity[0])  #Prints the cosine similarity values for the first row of the 'similarity' matrix.

print("===",ab['song'][3])  #Prints the value of the 'song' column at index 3 of the DataFrame 'ab'.

song_name = ab['song']


def recommendation(song):
    idx = ab[ab['song'] == song].index[0]
    distances = sorted(list(enumerate(similarity[idx])), reverse=True,   ## performance Optimization
                       key=lambda x: x[1])  ### 1st number song have similarity with its own and others...

    songs = []
    for i in distances[1:21]:
        songs.append(ab.iloc[i[0]].song)

    return songs

re = recommendation('Without Love')

print("--------------------")

print(re)


import pickle

with open('ab.pkl', 'wb') as f:
    pickle.dump(ab, f)
with open('ab.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

with open('similarity.pkl','wb') as f:
    pickle.dump(similarity, f)
with open('similarity.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

with open('song_name.pkl','wb') as f:
    pickle.dump(song_name, f)
with open('song_name.pkl', 'rb') as f:
    loaded_model = pickle.load(f)