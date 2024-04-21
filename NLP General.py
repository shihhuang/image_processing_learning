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

# # Set up
#
# Please follow the instructions from README.md to set up the virtual environment. Make sure when you run the notebook you are using the kernel you set up which uses the virtual environment.

# ## Import packages

# +
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle, argparse, os
import data_analysis as da
import shutil, traceback

# data processing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, classification_report

# Hugging Face
import torch
from transformers import Trainer, TrainingArguments
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, BertModel
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertConfig

from torch.cuda.amp import autocast, GradScaler
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, TensorDataset
