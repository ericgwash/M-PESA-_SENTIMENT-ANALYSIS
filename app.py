import streamlit as st
from tensorflow.keras.models import load_model
from keras.preprocessing.text import tokenizer_from_json
import json
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import asyncio

# Load the saved model
model = load_model('model.h5')
max_tweet_length = 50

# Load the tokenizer from file
with open('tokenizer.json') as f:
    data1 = f.read()
    tokenizer = tokenizer_from_json(data1)

# Define the lemmatizing function
def lemmatizer(text):
    wnl = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Tokenize the input text
    tokens = word_tokenize(text.lower())

    # Remove stopwords and lemmatize the remaining words
    lemmas = [wnl.lemmatize(word, pos=str2wordnet(pos)) for word, 
              pos in nltk.pos_tag(tokens) if word not in stop_words]

    # Combine the lemmatized words back into a single string
    lemmatized_text = " ".join(lemmas)

    return lemmatized_text

# Define the sentiment prediction function
async def predict_sentiment(input_text):
    # Preprocess the input text
    input_text = lemmatizer(input_text)
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_pad = pad_sequences(input_seq, maxlen=max_tweet_length, 
                              padding='post', truncating='post')

    # Make a prediction with the model
    prediction = model.predict(input_pad)

    # Define the mapping of labels to sentiment names
    labels = {
        0: "neutral",
        1: "negative",
        2: "positive"
    }

    # Convert the prediction to a label
    predicted_label = labels[np.argmax(prediction)]

    # Return the predicted sentiment label
    return predicted_label

# Define the Streamlit app
async def app():
    # Set the title of the app
    st.title("Sentiment Analysis App")

    # Add a text input field
    input_text = await st.text_input("Enter some text:")

    # Add a button to trigger the sentiment prediction
    if await st.button("Predict"):
        predicted_sentiment = await predict_sentiment(input_text)
        await st.write(f"The predicted sentiment is {predicted_sentiment}.")

if __name__ == '__main__':
    asyncio.run(app())



