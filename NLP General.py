# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: NLP Learning
#     language: python
#     name: nlp_learning
# ---

# # **Set up**
#
# Please follow the instructions from README.md to set up the virtual environment. Make sure when you run the notebook you are using the kernel you set up which uses the virtual environment.
#
# Learning materials:
# * Tensorflow: <https://www.youtube.com/playlist?list=PLQY2H8rRoyvzDbLUZkbudP-MFQZwNmU4S>
# * Explanation: <https://www.youtube.com/watch?v=GGLr-TtKguA&t=2275s>

# ## **Import packages**

# +
import pandas as pd
import numpy as np
import json

import matplotlib.pyplot as plt
from nltk.probability import FreqDist
import tensorflow as tf
# from tqdm import tqdm
# import pickle, argparse, os
# import data_analysis as da
# import shutil, traceback

# # data processing
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MultiLabelBinarizer
# from sklearn.metrics import accuracy_score, classification_report

# # Hugging Face
# import torch
# from transformers import Trainer, TrainingArguments
# from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, BertModel
# from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertConfig

# from torch.cuda.amp import autocast, GradScaler
# from torch.nn.functional import softmax
# from torch.utils.data import DataLoader, TensorDataset

import seaborn as sns
# import re
# from pandasql import sqldf
# import os
# import textwrap
# from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap




# +
import nltk
from nltk.corpus import stopwords
 
nltk.download('stopwords')
common_stopwords = stopwords.words('english')
print(common_stopwords)
# -

# ## **Datasets**

datastore = []
with open("./Sarcasm_Headlines_Dataset.json/Sarcasm_Headlines_Dataset.json", "r") as f:
    for line in f:
        datastore.append(json.loads(line))


# +
sentences = []
labels = []

for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
# -

print("Number of sentences: ", str(len(sentences)))
print("Number of labels: ", str(len(labels)))

# # **Explorative Data Analysis**

sentences_df = pd.DataFrame({"headline":sentences,
                             "is_sarcastic":labels
                            })
sentences_df.head()

# **Here we take a look at the overall percentage of sarcastic headlines. In this dataset, the distribution of 1 and 0 is relatively even. Here we do not have imbalanced dataset problem.**

overview_df = sentences_df.groupby("is_sarcastic").agg({"headline":"count"}).reset_index(drop=False)
overview_df['percentage'] = round(overview_df['headline']*100/overview_df['headline'].sum(),1)
overview_df

# ## **Number of words**

# Here we take a look at the number of words in each sentence (can think of it as "document").
#
# We also explore if the number of words have different distributions between Sarcastic and Non Sarcastic headlines.

sentences_df["number_of_words"] = sentences_df['headline'].apply(lambda x: len(x.split()))

sentences_df["number_of_words"].hist()

# +
# sentences_df.hist("number_of_words", by ="is_sarcastic" )
# -

sentences_df["number_of_words"].describe()

sentences_df["number_of_words"].max()

# +
plt.hist(sentences_df[sentences_df["is_sarcastic"]==0]["number_of_words"], alpha=0.5, label='Non Sarcastic')
plt.hist(sentences_df[sentences_df["is_sarcastic"]==1]["number_of_words"], alpha=0.5, label='Sarcastic')
#add plot title and axis labels
plt.title('Number of words by Sarcastic label')
plt.xlabel('Number of words')
plt.ylabel('Frequency')

#add legend
plt.legend(title='Label')

#display plot
plt.show()


# -

# ## **Top appearing words**

def extract_top_n_words(title_list, n=10):
    fdist = FreqDist(title_list)
    top_n = fdist.most_common(n)
    return top_n


all_words_list = [xx for x in sentences for xx in x.split() if xx not in common_stopwords]
all_top_n_words = extract_top_n_words(all_words_list)
all_top_n_words

num_unique_words = len(set(all_words_list))
num_unique_words

# **We check if the top appearing words could be different for different labels**

sarcastic_headlines = list(sentences_df[sentences_df['is_sarcastic']==1]['headline'])
non_sarcastic_headlines = list(sentences_df[sentences_df['is_sarcastic']==0]['headline'])

sarcastic_words_list = [xx for x in sarcastic_headlines for xx in x.split() if xx not in common_stopwords]
sarcastic_top_n_words = extract_top_n_words(sarcastic_words_list)
sarcastic_top_n_words

non_sarcastic_words_list = [xx for x in non_sarcastic_headlines for xx in x.split() if xx not in common_stopwords]
non_sarcastic_top_n_words = extract_top_n_words(non_sarcastic_words_list)
non_sarcastic_top_n_words

group_by = 'is_sarcastic'
summary_key_words_by_label = pd.DataFrame()
labels_plot = list(sentences_df[group_by].unique())
for label in labels_plot:
    headline_label_list = list(sentences_df[sentences_df[group_by] == label]['headline'])
    headline_label_list_words = [xx for x in headline_label_list for xx in x.split() if xx not in common_stopwords]
    top_n_words = extract_top_n_words(headline_label_list_words)
    for i in range(len(top_n_words)):
        new_row = pd.DataFrame({group_by: [label],
                                "word": [top_n_words[i][0]],
                                "count": [top_n_words[i][1]]})
        summary_key_words_by_label = pd.concat([summary_key_words_by_label, new_row])

pivot_df = summary_key_words_by_label.pivot(index="word", columns=group_by, values="count")
# Create a custom colormap
colors = ["#00b0cc", "grey", "#FF5800"]  # Define the colors
n_bins = 100  # Increase this number for a smoother transition
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=n_bins)
sns.heatmap(pivot_df, annot=True, fmt='.0f', cmap=cmap)
plt.title("Word Count by " + group_by)

# From the plot above we can see for example for headlines being Sarcastic, the distinct words include "area", "day", "man", "report:", "still", "time", "woman". However do note that words "woman" and "women" share the same context. There's another observation that "trump" appears often in the headlines. This triggers the question about whether or not we should remove these top appearing words in general as they could appear in headlines quite often, they could be ineffective in determining whether or not the headline is sarcastic. 

# # Data Processing

# ## Descriptions
#
# **In general, we need to clean the text before we make it into tokens. Common steps include:**
# - Lower case
# - Remove trailing or leading spaces
# - Remove hyper links
# - Remove punctuations (?)
#
# **Further, the words are tokenized into numbers. Basically this step is to convert the data into a format that the model understand - i.e. numerical format**

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ## Train and test split
#
# We don't want the model to see anything from the test set when it is training. If we really want to test its effectiveness - make sure the neural net only sees the training data. **Therefore** we split into train and test before tokenizing.

# +
train_ratio = 0.8
training_size = int(round(len(sentences)*train_ratio,0))
print(training_size)

vocab_size = 10000
embedding_dim = 16
max_length = 40
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"

# +
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]

training_labels = labels[0:training_size]
testing_labels = labels[training_size:]
# -

print("Training obs:", len(training_sentences))
print("Testing obs:", len(testing_sentences))
print("Total obs:", len(training_sentences)+len(testing_sentences))
print("Total obs from original:", len(sentences))

# ## Tokenizing with tokenizer 
#
# - Initiate tokenizer with *Tokenizer*
# - Use method *fit_on_texts* to convert words into indexes, this is like creating a dictionary
# - Then convert the training_sentences into training sequences (number format) using *tokenizer.texts_to_sequences()*
# - Also pad the sequences using *pad_sequences()*

tokenizer_all = Tokenizer(oov_token=oov_tok)
tokenizer_all.fit_on_texts(sentences)
word_index_all = tokenizer_all.word_index

tokenizer = Tokenizer(oov_token=oov_tok)

tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

print("Number of words tokenized:", len(word_index)-1)
print("Number of original words tokenized", len(word_index_all)-1)

# +
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, padding=padding_type, 
                                truncating=trunc_type, maxlen=max_length)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, padding=padding_type, 
                               truncating=trunc_type, maxlen=max_length)



training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

# -

training_labels

# **Note that we have chosen "post" padding, meaning adding 0's to the end of the sentence if it is less than 40 characters. As shown above, the maximum number of words in the headlines is 40 and therefore headlines less than 40 words will be padded. QUESTION: is it because we have added OOV and so the length increased from 39 to 40?**
#
# See documentation here: <https://www.tensorflow.org/api_docs/python/tf/keras/utils/pad_sequences>

training_sentences[1]

training_sequences[0]

training_padded[0]

print("Number of unique words including <OOV> in training:", str(len(word_index)))
print("Number of observations with the number of 'features' after padding in training:", training_padded.shape)

# ## Embeddings
# - This step is to understand the context and relationships between words. Words that only appear in the Sarcastic headlines would have a strong components in the Sarcastic direction.
# - The neural net can learn from the training process which words are more on the Sarcastic side and which are not
# - After the full training, the values are "added" together

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim), # Learning the meaning and context epoch by epoch
    tf.keras.layers.GlobalAveragePooling1D(), # Sum up the vectors - adding the vectors
    tf.keras.layers.Dense(24, activation='relu'), # activation 
    tf.keras.layers.Dense(1, activation='sigmoid'),
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

num_epochs = 30
history = model.fit(training_padded, training_labels, epochs=num_epochs,
                    validation_data=(testing_padded, testing_labels), verbose=2)

model.summary()


# +
def plot_graphs(history, string):
    plt.figure(figsize=(8,6))

    total_epochs = len(history.history[string])
    epochs = [i for i in range(total_epochs)]
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])

    # Add data labels to each point
    for i, j in zip(epochs, history.history[string]):
        plt.text(i, j*0.99, f'{round(j,2)}', ha='center', va='center',fontsize=8,rotation=45)

    # Add data labels to each point
    for i, j in zip(epochs, history.history['val_'+string]):
        plt.text(i, j*0.99, f'{round(j,2)}', ha='center', va='center',fontsize=8,rotation=45)
    
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")
# -

new_sentences = ["granny starting to fear spiders in the garden might be real",
            "game of thrones season finale showing this sunday night"]
new_sequences = tokenizer.texts_to_sequences(new_sentences)
new_padded = pad_sequences(new_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
print(model.predict(new_padded))

# # Recurrent Neural Network

# - Understanding sequences - we can think of the model as taking f(Data, Label) which is the rule -> think of the $n_x=n_{x-1} + n_{x-2}$
# - The numerical values can remain in the sequence for example
# - The recurrent neuron takes values sequentially while remaining the outputs from previous neurons
# - This weakens as time goes by -> helpful when required context is closeby
# - There could be problems where we need longer memory



# # Long Short Term Memory (LSTM)
# * An architextual that has "context" which can bring the meaning from the beginning of the sentence
# * It could be bidirectional! Words from future can give you context of current word. For more leanrning <deeplearning.ai >

# +
lstm_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim), # Learning the meaning and context epoch by epoch
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)), # Sum up the vectors - adding the vectors
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(24, activation='relu'), # activation 
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# -

lstm_model.summary()

num_epochs = 10
lstm_history = lstm_model.fit(training_padded, training_labels, epochs=num_epochs,
                        validation_data=(testing_padded, testing_labels), verbose=2)

lstm_model.summary()

plot_graphs(lstm_history, "accuracy")
plot_graphs(history, "accuracy")
