import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

import string

import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


df = pd.read_csv("emails.csv")


# NLP

nltk.download("punkt")
nltk.download("stopwords")

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

def filter_text(text: str) -> str:
    """
    This func gets an email text and filters, cleans it. 
    It deletes all stopwords and stems every word in the text.
    """
    
    # delete all symbols except ASCII letters
    # transform letters to lowercase
    ascii_text = ""
    for letter in text:
        if letter in string.ascii_letters + " ": 
            ascii_text += letter.lower()
    
    list_words = word_tokenize(ascii_text)

    # remove stopwords and stem every word
    filtered_words = [ ps.stem(word) 
                       for word in list_words 
                       if word not in stop_words 
                     ]
    
    return " ".join(filtered_words)

df["Message" ] = df["Message" ].apply(filter_text)
df["Category"] = df["Category"].apply(
    lambda category: 1 if category == "ham" else 0
)


# Feature extraction

cv = CountVectorizer()

# This method gets an iterable object containing strings
# It builds a matrix where count of rows are equal count of elements (strings) in the given iterable object
# Count of the matrix columns are equal count of all unique words found in this object
# The intersection of the column and the row is the number of occurrences of the word in the string

# Hence, for each email (a row of the matrix), we have a vector of numbers (the number of occurrences of words) 
# We will pass this vector (the features of the email) to the input of the logistic regression model

X = cv.fit_transform(df["Message"].values)
y = df["Category"].values


# Training the logistic regression model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# Result

general_count = len(y_test)
count_right_asns = 0

for i in range(general_count):
    if y_pred[i] == y_test[i]:
        count_right_asns += 1

ratio = round( count_right_asns / general_count, 2 )
print(f"Right answers: {count_right_asns}/{general_count} = {ratio}")
    

def isSpam(email_text: str) -> bool:
    """
    This func gets an email text (in english) and
    uses the trained model to determine whether 
    this email is spam or ham
    """
    
    filtered_text = filter_text(email_text)
    
    X = cv.transform([ filtered_text ])
    y = model.predict(X)
    
    return y[0] == 0
