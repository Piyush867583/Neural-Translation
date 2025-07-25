# Installation
"""

 !pip install transformers==4.31.0

 !pip install transformers datasets evaluate

 !pip install sacrebleu

"""# Imports"""

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
import evaluate
from numpy import random
import gensim.downloader as api
from PIL import Image
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense,Flatten,InputLayer,BatchNormalization,Dropout,Input,LayerNormalization
from tensorflow.keras.losses import BinaryCrossentropy,CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy,TopKCategoricalAccuracy, CategoricalAccuracy, SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from google.colab import drive
from google.colab import files
from datasets import load_dataset
from transformers import (DataCollatorWithPadding,create_optimizer,AutoTokenizer,DataCollatorForSeq2Seq,
                          T5TokenizerFast,T5ForConditionalGeneration,TFAutoModelForSeq2SeqLM)

BATCH_SIZE=64

# """# Data Preparation for Bert Model"""

 !wget https://www.manythings.org/anki/fra-eng.zip

 !unzip "/content/fra-eng.zip" -d "/content/dataset/"

dataset=load_dataset('text', data_files='/content/dataset/fra.txt')

dataset

dataset['train'][100000]['text'].split('\t')[:-1]

model_id="t5-small"
tokenizer=T5TokenizerFast.from_pretrained(model_id)

prefix = "translate English to French: "

def preprocess_function(examples):

  inputs = [prefix + example.split('\t')[0] for example in examples['text']]
  targets = [example.split('\t')[1] for example in examples['text']]

  model_inputs = tokenizer(inputs, text_target=targets,max_length=128, truncation=True)
  return model_inputs

tokenized_dataset=dataset.map(preprocess_function,batched=True)

tokenized_dataset

tokenized_dataset['train'][1000]

cleaned_dataset = tokenized_dataset.remove_columns("text")

cleaned_dataset['train'][1000]

model = TFAutoModelForSeq2SeqLM.from_pretrained(model_id)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,model=model, return_tensors="tf")

tf_train_set = cleaned_dataset["train"].to_tf_dataset(
    shuffle=True,
    batch_size=BATCH_SIZE,
    collate_fn=data_collator,
)

train_data=tf_train_set.take(int(0.9*len(tf_train_set)))
val_data=tf_train_set.skip(int(0.9*len(tf_train_set)))

for i in val_data.take(1):
  print(i)

"""# Modeling"""

model.summary()

"""# Training"""

num_epochs = 3
num_train_steps = len(train_data) * num_epochs

optimizer, schedule = create_optimizer(
    init_lr=2e-5,
    num_warmup_steps=0,
    num_train_steps=num_train_steps,
)
model.compile(optimizer=optimizer)

model.fit(
  train_data,
  validation_data=val_data,
  epochs=3
)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model_loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

"""# Evaluation"""

metric = evaluate.load("sacrebleu")

all_preds = []
all_labels = []

for batch in val_data:
  predictions = model.generate(
      input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
  )
  decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
  labels = batch["labels"].numpy()
  labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
  decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
  all_preds.extend(decoded_preds)
  all_labels.extend(decoded_labels)

result = metric.compute(predictions=all_preds, references=all_labels)
print(result)

decoded_preds

decoded_labels



"""# Testing

"""

input_text="Have you ever played soccer under the rain, with your friends? "
input_text="Google Translate is a multilingual neural machine translation service developed by Google to translate text, documents and websites from one language into another."
input_text=prefix + input_text
tokenized=tokenizer([input_text], return_tensors='tf')
out = model.generate(**tokenized, max_length=128)
print(out)

print(tokenizer.decode(out[0], skip_special_tokens=True))

"""# Testing Original Model (No FineTuning)"""

original_model = TFAutoModelForSeq2SeqLM.from_pretrained(model_id)

input_text="Have you ever played soccer under the rain, with your friends? "
#input_text="Google Translate is a multilingual neural machine translation service developed by Google to translate text, documents and websites from one language into another."
input_text=prefix + input_text
tokenized=tokenizer([input_text], return_tensors='tf')
out = original_model.generate(**tokenized, max_length=128)
print(out)

print(tokenizer.decode(out[0], skip_special_tokens=True))

