import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import pandas as pd
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
st.title("Predict Name App")

if st.checkbox("Full code"):
    st.text('''

# Data preprocessing

def normalizeArabic(text):
    text = text.strip()
    text = re.sub("[إأٱآا]", "ا", str(text))
    text = re.sub("ى", "ي", str(text))
    text = re.sub("ؤ", "ء", str(text))
    text = re.sub("ئ", "ء", str(text))
    text = re.sub("ة", "ه", str(text))
    return text
final_data['Clean_name']=final_data['Full_name'].apply(lambda x:normalizeArabic(x))

# Define the input and the target
x=final_data['Clean_name']
y=final_data['Sentence']
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# Encode dataset
tokenizer = Tokenizer()
tokenizer.fit_on_texts(final_data['Clean_name']) 
tokenized = tokenizer.texts_to_sequences( final_data['Clean_name'] ) 
input_word_dict = tokenizer.word_index
num_input_tokens = len( input_word_dict )+1
print( 'Number of input tokens = {}'.format( num_input_tokens))
max_input_length= max(len(seq) for seq in tokenized)
print('max_input_length: ', max_input_length)

# Train data

tokenizer_train =tokenizer
tokenized_train = tokenizer_train.texts_to_sequences(X_train) 
padded_train =pad_sequences( tokenized_train , maxlen=5 , padding='post' )
encode_train= np.array(padded_train)
print( 'Encoder input data shape -> {}'.format( encode_train.shape ))

# Test data

tokenizer_test = tokenizer
tokenized_test = tokenizer_test.texts_to_sequences(X_test) 
padded_test =pad_sequences( tokenized_test , maxlen=5 , padding='post' )
encode_test= np.array(padded_test)
print( 'Encoder input data shape -> {}'.format( encode_test.shape ))

# Define model

model = Sequential()
model.add(Embedding(num_input_tokens,10,
        input_length=5
    ))
model.add(LSTM(10))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
callbacks = [
    EarlyStopping(monitor='val_accuracy',
                  patience=3,
                  restore_best_weights=True,
                  verbose=1),
] # to avoid the overfitting

history = model.fit(x=encode_train,
                    y=y_train,
                    batch_size=16,
                    epochs=10,
                    validation_data=(encode_test, y_test),
                    callbacks=callbacks)

# save model
model.save('final_model.h5')
# Plot accuracies
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

y_predict=model.predict(encode_test)
precision, recall, thresholds = precision_recall_curve(y_test, y_predict)


# Define threshold
# I plot the output for percision and recal to choose what is the true value for threshold, so i choose the value that give the high precision without affecting Recall.
# Plot the output.
precision, recall, thresholds = precision_recall_curve(y_test, y_predict)
plt.plot(thresholds, precision[:-1], c ='r', label ='PRECISION')
plt.plot(thresholds, recall[:-1], c ='b', label ='RECALL')
plt.grid()
plt.legend()
plt.title('Precision-Recall Curve')
# The threshold value is around 0.71 which would increase Precision but not much decrease in Recall. 
# For each name of predict if the probability <threshold return incorrect for fake names else return correct
name=[]
for i in y_predict:
    if i<0.71:
        name.append('incorrect')
    else:
        name.append('correct') 
my_array = np.array(name)

# # Api
''')
if st.checkbox("Server app code"):
    st.text(''' 
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
page_bg_img = 
<Style>
[data-testid="stAppViewContainer"]{

background-image: url(https://falakstartups.com/wp-content/uploads/2019/12/Digified-logo.png..);    
background-size: contain ;
background-position: bottom;
background-repeat: no-repeat;
box-sizing: border-box;       
}
</Style>




def main():
    st.title("Predict Name")
    st.markdown(page_bg_img  , unsafe_allow_html = True)
    namey = st.text_input("Enter your name" ,"")
    if st.button("Predict"):
        result = name(namey)
        st.success("The name is {}".format(result))    
    
    
    
    
if __name__ == '__main__':
    main()
'''
           )


# In[ ]:


page_bg_img = '''
<Style>
[data-testid="stAppViewContainer"]{

background-image: url(https://falakstartups.com/wp-content/uploads/2019/12/Digified-logo.png..);    
background-size: contain ;
background-position: bottom;
background-repeat: no-repeat;
box-sizing: border-box;       
}
</Style>
'''



def main():
    st.markdown(page_bg_img  , unsafe_allow_html = True)
    namey = st.text_input("Enter your name" ,"")
    if st.button("Predict"):
        result = name(namey)
        st.success("The name is {}".format(result))    
    
    
    
    
if __name__ == '__main__':
    main()

