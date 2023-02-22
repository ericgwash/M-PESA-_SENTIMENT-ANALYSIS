import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

# Load the tokenizer
with open("tokenizer.json", "r") as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

# Load the model
model = tf.keras.models.load_model('model.h5')

# Define a function to preprocess text before making predictions
def preprocess_text(text):
    # Tokenize the text
    text = tokenizer.texts_to_sequences([text])
    # Pad the sequences
    text = tf.keras.preprocessing.sequence.pad_sequences(text, maxlen=100, padding='post', truncating='post')
    return text

# Define the app
def app():
    # Set the app title
    st.set_page_config(page_title='LSTM Model App')

    # Add a title
    st.title('LSTM Model App')

    # Add a description
    st.write('Enter some text and the model will predict the next word.')

    # Add a text input
    input_text = st.text_input('Enter some text:')

    # Add a button to make predictions
    if st.button('Predict'):
        # Preprocess the input text
        input_text = preprocess_text(input_text)

        # Make a prediction
        prediction = model.predict(input_text)

        # Get the index of the predicted word
        index = tf.argmax(prediction, axis=1)[0].numpy()

        # Get the predicted word from the tokenizer
        predicted_word = tokenizer.index_word[index]

        # Display the predicted word
        st.write('The predicted next word is:', predicted_word)

# Run the app
if __name__ == '__main__':
    app()


