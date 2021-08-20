import keras 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
import streamlit as st

text = open(r'C:\Users\ASUS\LGM_next_char\1661-0.txt', encoding='utf-8').read().lower()
model = keras.models.load_model(r"C:\Users\ASUS\LGM_next_char\Model\my_model_character_generation.h5")

tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(text)
sequences = tokenizer.texts_to_sequences(text)
maxid = len(tokenizer.word_index)

def preprocess(text):
  X = np.array(tokenizer.texts_to_sequences(text))-1
  return tf.one_hot(X,maxid)

def next_char(text, temperature=1):
    X_new = preprocess([text])
    y_proba = model(X_new)[0, -1:, :]
    rescaled_logits = tf.math.log(y_proba) / temperature
    char_id = tf.random.categorical(rescaled_logits, num_samples=1) + 1
    return tokenizer.sequences_to_texts(char_id.numpy())[0] 

def complete_text(text, n_chars=10, temperature=1):
    for _ in range(n_chars):
        text += next_char(text, temperature)
    return text

def main():
    st.title('Text generator')
    tempearature = st.sidebar.slider('Select Temperature',0.0, 2.0, (0.75))
    print(tempearature)
    n_chars = st.sidebar.slider('Select No of Character which you wants to predict',1, 60, (50))
    print(n_chars)
    sentence = st.text_input('Input your sentence here:')
    if st.button('Generate Text'):
        if sentence:
            st.write(complete_text(sentence,temperature=tempearature,n_chars=n_chars))
        else:
            st.warning('Please enter something in the text box')
if __name__ == "__main__":
    main()

    
