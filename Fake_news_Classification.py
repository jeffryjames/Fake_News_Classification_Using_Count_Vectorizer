import pandas as pd

data = pd.read_csv("D:/ML/Fake News Classification/train.csv")
print(data.head())
print(data.columns)
print(data.shape)

X = data.drop("label", axis = 1)
print(X)

y = data["label"]
print(y)


data = data.dropna()
data_copy = data

data.reset_index(inplace = True)

data["title"][0]

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
ps = PorterStemmer()
corpus = []

for i in range(0, len(data)):
    review = re.sub('[^a-zA-Z]', ' ' , data['text'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    print(i)
    

import pickle
corpus_copy = corpus
pickle.dump(corpus_copy, open('model.pkl','wb'))
corpus_model = pickle.load(open('model.pkl','rb'))

corpus_model[0]

# Applying Countvectorizer
# Creating the Bag of Words model

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000, ngram_range=(1,3))
X = cv.fit_transform(corpus_copy).toarray()

print(X.shape)

y = data["label"]
print(y)

# Divide the dataset into Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

cv.get_feature_names()[:20]
cv.get_params()
count_df = pd.DataFrame(X_train, columns=cv.get_feature_names())
count_df.head()

import matplotlib.pyplot as plt
import numpy as np
import itertools

#function code from scikit html

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
   
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
    
## MultinomialNB Algorithm
# Multinomial Naive Bayesfor text data works good
from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()
classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)

from sklearn import metrics
score = metrics.accuracy_score(y_test, pred)
percent = score*100
print("accuracy:   %0.3f" % percent)
cm = metrics.confusion_matrix(y_test, pred)
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])

classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
score





    