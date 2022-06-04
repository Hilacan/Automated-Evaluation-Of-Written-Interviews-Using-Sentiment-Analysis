from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

model = load_model('model/A-model.h5')

ans = ["I want to work in CIT-University because I want to serve my alma mater as a return of gratitude to the institution; that I graduated or finished my education in CIT-University for 17 years, from elementary to college. If it wasn't for CIT-University, my siblings and I wouldn't have the opportunity to finish our education in a prestigious institution, where they offered privilege program for students whose parents' are employees of CIT-University, where my mother is also currently working as a teacher and where my grandfather was a former Registrar of High School Department."]
def predict(response):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(response)
    sequenced = tokenizer.texts_to_sequences(response)
    padded = pad_sequences(sequenced, maxlen=74, dtype='int32', value=0)
    sentiment = model.predict(padded,batch_size=4,verbose = 2)[0]
    for i in sentiment:
       val = i
    return val+.2

def label(val):
    if val >= 0.5:
        label = "Well Done! You have passed the evaluation!"
    else:
        label = "Unfortunately, you did not pass"
    return label
print(predict(ans))
print(label(predict(ans)))













