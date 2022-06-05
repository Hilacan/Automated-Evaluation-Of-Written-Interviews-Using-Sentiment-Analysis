from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

model = load_model('model/model-c.h5')
model1 = load_model('model/A-Model.h5')


answer = "Ever since I was young it had always been my passion to teach and mold future"

ans = [answer]

def predict(response):
    max_fatures = 500
    tokenizer = Tokenizer(num_words=max_fatures, split=' ')
    tokenizer.fit_on_texts(response)
    sequenced = tokenizer.texts_to_sequences(response)
    padded = pad_sequences(sequenced, maxlen=71, dtype='int32', value=0)
    sentiment = model.predict(padded,batch_size=4,verbose = 2)[0]
    for i in sentiment:
       val = i
    return val

def label(val):
    if val >= 0.5:
        label = "Well Done! You have passed the evaluation!"
    else:
        label = "Unfortunately, you did not pass"
    return label

# a=predict(ans)
# print(a)
# print(label(a))













