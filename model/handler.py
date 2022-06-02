from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
model = load_model('model/SA_LSTM.h5')

model.summary()
answer = [" During the initial interview, i can already tell this is the type of environment I'm excited to work in and thrive in in the long term. The opportunities to tackle different projects are what draw me since I’m inclined to be solution-oriented. The opportunities to expand in this field are something that cannot be grasp anywhere else.  It's challenging and never redundant. I'm thrilled by the idea of working on projects in an environment I'm quite familiar with and working with people i look up to. For me, this opportunity offers an environment for me to expand professionally."]

# def prediction(answer):

#     tokenizer = Tokenizer()
#     tokenizer.fit_on_texts(answer)
#     sequenced = tokenizer.texts_to_sequences(answer)
#     padded = pad_sequences(sequenced, maxlen=74, dtype='int32', value=0)
#     sentiment = model.predict(padded,batch_size=1,verbose = 2)[0]
#     eval = "NOT HIRED"
#     if (np.argmax(sentiment) == 1):
#         eval = "HIRED"
#     return eval

tokenizer = Tokenizer()
tokenizer.fit_on_texts(answer)
sequenced = tokenizer.texts_to_sequences(answer)
padded = pad_sequences(sequenced, maxlen=74, dtype='int32', value=0)
sentiment = model.predict(padded,batch_size=1,verbose = 2)[0]
eval = "NOT HIRED"
if (np.argmax(sentiment) == 1):
     eval = "HIRED"





