
import tensorflow as tf### models
import numpy as np### math computations
import matplotlib.pyplot as plt### plotting bar chart
import sklearn### machine learning library
import cv2## image processing
from sklearn.metrics import confusion_matrix, roc_curve### metrics
import seaborn as sns### visualizations
import datetime
import pathlib
import io
import os
import re
import string
import time
from numpy import random
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import (Dense,Flatten,SimpleRNN,InputLayer,Conv1D,Bidirectional,GRU,LSTM,BatchNormalization,Dropout,Input, Embedding,TextVectorization)
from tensorflow.keras.losses import BinaryCrossentropy,CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy,TopKCategoricalAccuracy, CategoricalAccuracy, SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from google.colab import drive
from google.colab import files
from tensorboard.plugins import projector

"""# Data Preparation

## Data Download
"""

!wget https://www.manythings.org/anki/fra-eng.zip

!unzip "/content/fra-eng.zip" -d "/content/dataset/"

"""## Kaggle Dataset"""

!pip install -q kaggle
!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 /root/.kaggle/kaggle.json
!kaggle datasets download -d dhruvildave/en-fr-translation-dataset

!unzip "/content/en-fr-translation-dataset.zip" -d "/content/dataset/"

dataset = tf.data.experimental.CsvDataset(
  "/content/dataset/en-fr.csv",
  [
    tf.string,
    tf.string
  ],
)

"""## Data Processing"""

text_dataset=tf.data.TextLineDataset("/content/dataset/fra.txt")

VOCAB_SIZE=20000
ENGLISH_SEQUENCE_LENGTH=64
FRENCH_SEQUENCE_LENGTH=64
EMBEDDING_DIM=300
BATCH_SIZE=64

english_vectorize_layer=TextVectorization(
    standardize='lower_and_strip_punctuation',
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=ENGLISH_SEQUENCE_LENGTH
)

french_vectorize_layer=TextVectorization(
    standardize='lower_and_strip_punctuation',
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=FRENCH_SEQUENCE_LENGTH
)

def selector(input_text):
  split_text=tf.strings.split(input_text,'\t')
  return {'input_1':split_text[0:1],'input_2':'starttoken '+split_text[1:2]},split_text[1:2]+' endtoken'

split_dataset=text_dataset.map(selector)

def separator(input_text):
  split_text=tf.strings.split(input_text,'\t')
  return split_text[0:1],'starttoken '+split_text[1:2]+' endtoken'

init_dataset=text_dataset.map(separator)

for i in split_dataset.take(3):
  print(i)

english_training_data=init_dataset.map(lambda x,y:x)### input x,y and output x
english_vectorize_layer.adapt(english_training_data)#### adapt the vectorize_layer to the training data

french_training_data=init_dataset.map(lambda x,y:y)### input x,y,z and output y
french_vectorize_layer.adapt(french_training_data)#### adapt the vectorize_layer to the training data

def vectorizer(inputs,output):
  return {'input_1':english_vectorize_layer(inputs['input_1']),
          'input_2':french_vectorize_layer(inputs['input_2'])},french_vectorize_layer(output)

split_dataset

dataset=split_dataset.map(vectorizer)

for i in split_dataset.take(3):
  print(i)

for i in dataset.take(1):
  print(i)

dataset

dataset=dataset.shuffle(2048).unbatch().batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

dataset

NUM_BATCHES=int(200000/BATCH_SIZE)

train_dataset=dataset.take(int(0.9*NUM_BATCHES))
val_dataset=dataset.skip(int(0.9*NUM_BATCHES))

train_dataset

"""# Modeling

## Encoder
"""

class Encoder(tf.keras.Model):
  def __init__(self,vocab_size,embedding_dim,units):
    super(Encoder,self).__init__()
    self.embedding_dim=embedding_dim
    self.vocab_size=vocab_size
    self.units=units

  def build(self,input_shape):
    self.embedding=Embedding(self.vocab_size,self.embedding_dim)
    self.lstm=LSTM(self.units,return_sequences=True)

  def call(self,x):
    x=self.embedding(x)
    output=self.lstm(x)
    return output

HIDDEN_UNITS=256
EMBEDDING_DIM=256
encoder=Encoder(VOCAB_SIZE,EMBEDDING_DIM,HIDDEN_UNITS)
encoder_output=encoder(tf.zeros([128,8]))
print(encoder_output.shape)

"""## Bahdanau Attention"""

class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self,units):
    super(BahdanauAttention,self).__init__()
    self.units=units

  def build(self,input_shape):
    self.w_1=tf.keras.layers.Dense(self.units)
    self.w_2=tf.keras.layers.Dense(self.units)
    self.w=tf.keras.layers.Dense(1)

  def call(self,prev_dec_state,enc_states):
    scores=self.w(
        tf.nn.tanh(
        self.w_1(tf.expand_dims(prev_dec_state,-2)) +
        self.w_2(enc_states)))

    attention_weights=tf.nn.softmax(scores,axis=1)
    context_vector=attention_weights*enc_states
    context_vector=tf.reduce_sum(context_vector,axis=1)
    return context_vector,attention_weights

bahdanau_attention=BahdanauAttention(256)
context_vector,attention_weights=bahdanau_attention(tf.zeros([128,32]),tf.zeros([128,8,32]))
print(context_vector.shape)
print(attention_weights.shape)

"""## Decoder"""

class Decoder(tf.keras.Model):
  def __init__(self,vocab_size,embedding_dim,dec_units,sequence_length):
    super(Decoder,self).__init__()
    self.embedding_dim=embedding_dim
    self.vocab_size=vocab_size
    self.dec_units=dec_units
    self.sequence_length=sequence_length

  def build(self,input_shape):
    self.dense=Dense(self.vocab_size,activation="softmax")
    self.gru=GRU(
        self.dec_units,return_sequences=True,return_state=True)
    self.attention=BahdanauAttention(self.dec_units)
    self.embedding=Embedding(self.vocab_size,self.embedding_dim)

  def call(self,x,hidden,shifted_target):
    outputs=[]
    context_vectors=[]
    attention_weightss=[]
    shifted_target=self.embedding(shifted_target)

    for t in range(0,self.sequence_length):
      context_vector,attention_weights=self.attention(hidden,x)
      dec_input=context_vector+shifted_target[:,t]
      output,hidden=self.gru(tf.expand_dims(dec_input,1))
      outputs.append(output[:,0])

    outputs=tf.convert_to_tensor(outputs)
    outputs=tf.transpose(outputs, perm=[1,0,2])

    outputs=self.dense(outputs)
    return outputs,attention_weights

decoder=Decoder(VOCAB_SIZE,EMBEDDING_DIM,HIDDEN_UNITS,FRENCH_SEQUENCE_LENGTH)
outputs,attention_weights=decoder(encoder_output,tf.zeros([128,HIDDEN_UNITS]),tf.zeros([128,64]))
print(outputs.shape)
print(attention_weights.shape)

### ENCODER
input = Input(shape=(ENGLISH_SEQUENCE_LENGTH,), dtype="int64", name="input_1")
encoder=Encoder(VOCAB_SIZE,EMBEDDING_DIM,HIDDEN_UNITS)
encoder_output=encoder(input)

### DECODER
shifted_target=Input(shape=(FRENCH_SEQUENCE_LENGTH,), dtype="int64", name="input_2")
decoder=Decoder(VOCAB_SIZE,EMBEDDING_DIM,HIDDEN_UNITS,FRENCH_SEQUENCE_LENGTH)
decoder_output,attention_weightss=decoder(encoder_output,tf.zeros([1,HIDDEN_UNITS]),shifted_target)

### OUTPUT
bahdanau=Model([input,shifted_target],decoder_output)
bahdanau.summary()



class BLEU(tf.keras.metrics.Metric):
    def __init__(self,name='bleu_score'):
        super(BLEU,self).__init__()
        self.bleu_score=0

    def update_state(self,y_true,y_pred,sample_weight=None):
      y_pred=tf.argmax(y_pred,-1)
      self.bleu_score=0
      for i,j in zip(y_pred,y_true):
        tf.autograph.experimental.set_loop_options()

        total_words=tf.math.count_nonzero(i)
        total_matches=0
        for word in i:
          if word==0:
            break
          for q in range(len(j)):
            if j[q]==0:
              break
            if word==j[q]:
              total_matches+=1
              j=tf.boolean_mask(j,[False if y==q else True for y in range(len(j))])
              break

        self.bleu_score+=total_matches/total_words

    def result(self):
        return self.bleu_score/BATCH_SIZE

bahdanau.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(5e-4),
    metrics=[BLEU()],
    run_eagerly=True)

checkpoint_filepath = '/content/drive/MyDrive/nlp/translation/bahdanau_attention_1.h5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

history=bahdanau.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=30,)
    #callbacks=[model_checkpoint_callback])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model_loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('model_accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

bahdanau.evaluate(val_dataset)

"""# Testing"""

index_to_word={x:y for x, y in zip(range(len(french_vectorize_layer.get_vocabulary())),
                                   french_vectorize_layer.get_vocabulary())}

def translator(english_sentence):
  tokenized_english_sentence=english_vectorize_layer([english_sentence])
  shifted_target='starttoken'

  for i in range(FRENCH_SEQUENCE_LENGTH):
    tokenized_shifted_target=french_vectorize_layer([shifted_target])
    output=bahdanau.predict([tokenized_english_sentence,tokenized_shifted_target])
    french_word_index=tf.argmax(output,axis=-1)[0][i].numpy()
    current_word=index_to_word[french_word_index]
    if current_word=='endtoken':
      break
    shifted_target+=' '+current_word
  return shifted_target[11:]

translator('What makes you think that is not true?')

translator('Have you ever watched a soccer under the rain?')

translator('Great trees do not grow with ease, the stronger the winds, the stronger the trees')

translator('Everyone should water his or her tomato plants')

word_to_index={y:x for x, y in zip(range(len(french_vectorize_layer.get_vocabulary())),
                                   french_vectorize_layer.get_vocabulary())}

