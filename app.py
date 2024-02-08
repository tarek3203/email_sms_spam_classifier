import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))


def transform_text(text):
    # lowercase
    text = text.lower()

    # tokenize
    text = nltk.word_tokenize(text)

    # removing special character

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    # removing stop words and punctuation

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    # stemming

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


st.title('Email/SMS Spam Classifier')


input_sms = st.text_input("Enter The Message")

if st.button('predict'):
    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])

    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. display
    if result == 1:
        st.header('Spam')
    else :
        st.header('Not Spam')