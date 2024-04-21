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

# ## **Import packages**

# +
import pandas as pd
import numpy as np
import json

import matplotlib.pyplot as plt
from nltk.probability import FreqDist
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
labels = list(sentences_df[group_by].unique())
for label in labels:
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

tokenizer = Tokenizer(oov_token="<OOV>")

tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post')

# **Note that we have chosen "post" padding, meaning adding 0's to the end of the sentence if it is less than 40 characters. As shown above, the maximum number of words in the headlines is 40 and therefore headlines less than 40 words will be padded. QUESTION: is it because we have added OOV and so the length increased from 39 to 40?**

padded[0]

print("Number of unique words including <OOV>:", str(len(word_index)))
print("Number of observations with the number of 'features' after padding:", padded.shape)
