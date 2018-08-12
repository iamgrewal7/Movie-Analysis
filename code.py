# Import required libraries
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

#Import required data
genre = sys.argv[1]
omdb = sys.argv[2]
rotten = sys.argv[3]
wiki = sys.argv[4]

dataWiki = pd.read_json(wiki, orient='record', lines=True)
dataGen = pd.read_json(genre, orient='record', lines=True)
dataRot = pd.read_json(rotten, orient='record', lines=True)
dataOM = pd.read_json(omdb, orient='record', lines=True)

# Clean data
#dataWiki = dataWiki.dropna()
dataRot = pd.merge(dataRot, dataOM, on = 'imdb_id', how = 'left')
dataRot = dataRot.dropna()
dataWiki['genre']= dataWiki['genre'].apply(lambda x: ''.join(x))
dataWiki = dataWiki.drop(['based_on','filming_location','made_profit','main_subject','series','metacritic_id'],axis=1)
#dataWiki = pd.merge(dataWiki, dataRot,on='rotten_tomatoes_id',how='left')

#--------------------------------NLP-------------------------------------

# Create new dataframe for plot/Genre/success classification
data = pd.DataFrame()

# Create column Rating and Plot with data from Rotten Tomatoes and Omdb
data['Rating'] = dataRot['audience_percent']
data['Plot']  = dataRot['omdb_plot']
data['Genre'] = dataRot['omdb_genres'].apply(lambda x: ' '.join(x))
data['Plot/Genre'] = data['Plot'] + data['Genre']


# Assuming movies with rating >=50% is successful while other are not 
data['Rating'] = data['Rating']>49

# ------------Using genre as independent variable
print("")
print("----------Using Genre as independent Variable----------")
print("Accuracy Scores of Classfiers =>")
# Get dependent and independent variable
X = data.iloc[:,1].values

#X = np.apply_along_axis(lambda x: ' '.join(x),1,X)
y = data.iloc[:,0].values

# Using CountVector to convert words into token and transforming data
vect = CountVectorizer(stop_words='english', ngram_range = (1,1), max_df = .80, min_df = 4)
X = vect.fit_transform(X)

#Split data
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1, test_size= 0.2) 

#SVM model
classifier = SVC(kernel = 'rbf')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("SVM :",accuracy_score(y_test,y_pred))

# Logistic Regression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("Logistic Regression :",accuracy_score(y_test,y_pred))

#KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("KNN :",accuracy_score(y_test,y_pred))

# Naive Bayes
classifier = MultinomialNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("Naive Bayes :",accuracy_score(y_test,y_pred))

# Random Forest
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
classifier.fit(X_train, y_train)
y_pred  = classifier.predict(X_test)
print("Random Forest :",accuracy_score(y_test,y_pred))

#--------------Using Plot as independent variable
print("")
print("----------Using Plot as independent Variable----------")
print("Accuracy Scores of Classfiers =>")
# Get dependent and independent variable
X = data.iloc[:,2].values

#X = np.apply_along_axis(lambda x: ' '.join(x),1,X)
y = data.iloc[:,0].values

# Using CountVector to convert words into token and transforming data
vect = CountVectorizer(stop_words='english', ngram_range = (1,1), max_df = .80, min_df = 4)
X = vect.fit_transform(X)

#Split data
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1, test_size= 0.2) 

#SVM model
classifier = SVC(kernel = 'rbf')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("SVM :",accuracy_score(y_test,y_pred))

# Logistic Regression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("Logistic Regression :",accuracy_score(y_test,y_pred))

#KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("KNN :",accuracy_score(y_test,y_pred))

# Naive Bayes
classifier = MultinomialNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("Naive Bayes :",accuracy_score(y_test,y_pred))

# Random Forest
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
classifier.fit(X_train, y_train)
y_pred  = classifier.predict(X_test)
print("Random Forest :",accuracy_score(y_test,y_pred))
#--------Using both plot and genre
print("")
print("-------Using both Genre and Plot as independent Variable-------")
print("Accuracy Scores of Classfiers =>") 

# Get dependent and independent variable
X = data.iloc[:,3].values
#X = np.apply_along_axis(lambda x: ' '.join(x),1,X)
y = data.iloc[:,0].values

# Using CountVector to convert words into token and transforming data
vect = CountVectorizer(stop_words='english', ngram_range = (1,1), max_df = .80, min_df = 4)
X = vect.fit_transform(X)

#Split data
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1, test_size= 0.2) 

#SVM model
classifier = SVC(kernel = 'rbf')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("SVM :",accuracy_score(y_test,y_pred))

# Logistic Regression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("Logistic Regression :",accuracy_score(y_test,y_pred))

#KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("KNN :",accuracy_score(y_test,y_pred))

# Naive Bayes
classifier = MultinomialNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("Naive Bayes :",accuracy_score(y_test,y_pred))

# Random Forest
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
classifier.fit(X_train, y_train)
y_pred  = classifier.predict(X_test)
print("Random Forest :",accuracy_score(y_test,y_pred))


#------------------------------------Cluster----------------------------
def to_float(x):
    return float(x)

dataWiki['publication_date'] = dataWiki['publication_date'].str.slice(start = 0,stop = 4)
dataWiki['publication_date'] = dataWiki['publication_date'].apply(lambda x: to_float(x))

df = pd.DataFrame()
df['Genre'] = dataWiki['genre']
df['Date']= dataWiki['publication_date']
df = df.dropna()
X = df.iloc[:,[0,1]].values

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])

from sklearn.cluster import KMeans

#Using the elbow method to find the optimal number of cluster
# Learned from Udemy
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++',random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(1)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS') 
plt.savefig("Optimal Number of Clusters.png")

kmeans = KMeans(n_clusters = 3, init = 'k-means++')
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.figure(2)
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s = 100, c = 'red', label = 'Careful')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s = 100, c = 'blue', label = 'Sensible')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s = 100, c = 'green', label = 'Target')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of Movie Genre')
plt.xlabel('Encoded Movie Genre')
plt.ylabel('Year')
plt.legend()
plt.savefig("Clusters")
print("")
print("Both Cluster plots saved")
print("")




