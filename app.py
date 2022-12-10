# !pip install streamlit
import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re

model=load_model('final_model.h5')
def clean_text(text):
    text = text.strip()
    text = re.sub("[إأٱآا]", "ا", str(text))
    text = re.sub("ى", "ي", str(text))
    text = re.sub("ؤ", "ء", str(text))
    text = re.sub("ئ", "ء", str(text))
    text = re.sub("ة", "ه", str(text))
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    input_word_dict = tokenizer.word_index
    tokenized = tokenizer.texts_to_sequences(text) 
    padded=pad_sequences( tokenized , maxlen=5 , padding='post' )
    return padded

def name(text):
    text = clean_text(text)
    pred = model.predict(text)
    if pred[0][0]<0.71:
            return 'incorrect'
    else:
            return 'correct'
page_bg_img = ''''
<Style>
[data-testid="stAppViewContainer"]{

background-image: url(https://images.wuzzuf-data.net/files/company_logo/Verified-Egypt-37566-1552211585-og.png..);    
background-size: cover ;
background-repeat: no-repeat;       
}
</Style>
'''



def main():
    st.title("Predict Name App")
    st.markdown(page_bg_img  , unsafe_allow_html = True)
    namey = st.text_input("Enter your name" ,"")
    if st.button("Predict"):
        result = name(namey)
        st.text("The output is {}".format(result))
    
    
    
    
    
if __name__ == '__main__':
    main()
