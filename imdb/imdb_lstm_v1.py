#!/usr/bin/env python
# coding: utf-8

# In[36]:


import sklearn
from sklearn import datasets
from sklearn import svm
import numpy as np
import pandas as pd
 


# In[37]:


imdb = np.load(r'C:\Users\dexter\Desktop\data\imdb_train.npz')
imdb_test = np.load(r'C:\Users\dexter\Desktop\data\imdb_test.npz')

x_train = imdb['x']
y_train = imdb['y']
x_test = imdb_test['x']
y_test = imdb_test['y']


# In[63]:


import warnings
warnings.filterwarnings('ignore')

# Modules for data manipulation
import numpy as np
import pandas as pd
import re

# Modules for visualization
import matplotlib.pyplot as plt
import seaborn as sb

# Tools for preprocessing input data
from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Tools for creating ngrams and vectorizing input data
from gensim.models import Word2Vec, Phrases

# Tools for building a model
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences

# Tools for assessing the quality of model prediction
from sklearn.metrics import accuracy_score, confusion_matrix


import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from importlib import reload
import sys
from imp import reload

from tqdm import tqdm


if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIG_SIZE = 16
LARGE_SIZE = 20

params = {
    'figure.figsize': (16, 8),
    'font.size': SMALL_SIZE,
    'xtick.labelsize': MEDIUM_SIZE,
    'ytick.labelsize': MEDIUM_SIZE,
    'legend.fontsize': BIG_SIZE,
    'figure.titlesize': LARGE_SIZE,
    'axes.titlesize': MEDIUM_SIZE,
    'axes.labelsize': BIG_SIZE
}
plt.rcParams.update(params)


# In[43]:


import nltk#安装导入词向量计算工具
stop_words = set(stopwords.words("english")) #停词
lemmatizer = WordNetLemmatizer()#提取单词的主干


# In[44]:


def clean_text(text):
    # 用正则表达式取出符合规范的部分
    text = re.sub(r'[^\w\s]','',text, re.UNICODE)

    ##小写化所有的词，并转成词list
    text = text.lower()

    ##第一个参数表示待处理单词，必须是小写的；第二个参数表示POS，默认为NOUN
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words]
    text = " ".join(text)
    return text


# In[74]:


df = pd.DataFrame({'x':imdb['x'], 'y':imdb['y']})

lis = []
pbar = tqdm(total=36000)
for i in df['x']:
    x = clean_text(i)
    lis.append(x)
    pbar.update(1)
pbar.close()


# In[75]:


df['Processed_Reviews'] = lis
df.head()

df.Processed_Reviews.apply(lambda x: len(x.split(" "))).mean()


# In[76]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras.layers import Convolution1D
from keras import initializers, regularizers, constraints, optimizers, layers

max_features = 6000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(df['Processed_Reviews'])
list_tokenized_train = tokenizer.texts_to_sequences(df['Processed_Reviews'])

maxlen = 130
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
y = df['y']

embed_size = 128
model = Sequential()
model.add(Embedding(max_features, embed_size))
model.add(Bidirectional(LSTM(64, return_sequences = True)))
model.add(GlobalMaxPool1D())
model.add(Dense(20, activation="relu"))
model.add(Dropout(0.05))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 100
epochs = 3
model.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.2)


# In[77]:


df_test = pd.DataFrame({'x':imdb_test['x'], 'y':imdb_test['y']})
df_test.head()
lis1 = []
pbar = tqdm(total=10000)
for i in df_test['x']:
    x = clean_text(i)
    lis1.append(x)
    pbar.update(1)
pbar.close()
df_test['review'] = lis1
df_test["sentiment"] = df_test["y"]
y_test = df_test["sentiment"]
list_sentences_test = df_test["review"]
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)
prediction = model.predict(X_te)
y_pred = (prediction > 0.5)
from sklearn.metrics import f1_score, confusion_matrix
print('F1-score: {0}'.format(f1_score(y_pred, y_test)))
print('Confusion matrix:')
confusion_matrix(y_pred, y_test)


# In[78]:


y_pred = model.predict(X_te)
def submit(predictions):
    df_test['sentiment'] = predictions
    df_test.to_csv(r'C:\Users\dexter\Desktop\data\submission.csv', columns=['sentiment'])

submit(y_pred)


# In[ ]:




