
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn import ensemble
from sklearn.model_selection import train_test_split
dataset=pd.read_csv('Project.csv',encoding = "ISO-8859-1")
print('Dataset Loaded')
pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',500)
pd.set_option('display.width',1000)
dataset.describe()
dataset.info()
dataset.describe(include = [np.number])#stats for numeric attributes
dataset.describe(include = ['O']) #frequent,unique values
plt.style.use('seaborn')##Histograms for numeric data
dataset.hist(figsize=[20,20]) 
plt.show()
scatterdata=dataset.drop(dataset.columns[[0,1,5,6,9,10,11,13,14,15,16,17,19]], axis=1)
scatterdata=dataset[['movie_facebook_likes','title_year','budget','gross','director_facebook_likes',
                  'actor_1_facebook_likes','imdb_score','num_critic_for_reviews','duration','num_voted_users']]

sns.set(style="ticks")
sns.pairplot(scatterdata)
cor=dataset.corr(method='pearson')## Correlation matrix with pearson method -1 negative , 1 positive correlation
cor.style.background_gradient(cmap='Purples')
#Deeper blue color highlights higher Pearson correlation
dataset['actors_facebook_likes']=dataset['actor_1_facebook_likes']+dataset['actor_2_facebook_likes']+dataset['actor_3_facebook_likes']
kmeansdata= dataset[['movie_facebook_likes','num_critic_for_reviews','num_voted_users','duration']]
n_clusters = list(range(2, 11))
scores = []
for i in n_clusters:
    model = KMeans(n_clusters=i,random_state=0)
    model.fit(kmeansdata)
    results=model.labels_
    scores.append(silhouette_score(kmeansdata,results))
model = KMeans(n_clusters=3,random_state=0)
model.fit(kmeansdata)
results=model.labels_
score=silhouette_score(kmeansdata,results)
print("For n_clusters = 3 the average silhouette_score is :", score)
results=pd.Series(data=results,index=dataset.index)
results=results.to_frame("clusters")
dataset=dataset.join(results)    
plt.ylabel('Population',fontsize=16)
plt.xlabel('Clusters',fontsize=16)
dataset['clusters'].hist(figsize=(10,5))
plt.show()

plt.style.use('seaborn')
dataset[['actors_facebook_likes','movie_facebook_likes','director_facebook_likes']].groupby(dataset['clusters']).mean().plot.bar(stacked=False,figsize=(12,7))
plt.xlabel('Clusters',fontsize=15)
plt.legend(loc=0, prop={'size': 15})
plt.title('Mean values of movie,actors and director facebook likes per cluster',fontsize=18)
plt.show()

plt.style.use('seaborn')
dataset[['budget','gross']].groupby(dataset['clusters']).mean().plot.bar(stacked=False,figsize=(10,6))
plt.legend(loc=0, prop={'size': 14})
plt.xlabel('Clusters',fontsize=15)
plt.title('Mean value of budget and gross revenue of movies per cluster',fontsize=18)
plt.show()

sorted_fl=dataset.sort_values(by="movie_facebook_likes",ascending=False)
top10fl=sorted_fl.head(10)
top10fl[['movie_facebook_likes']].groupby(top10fl['movie_title']).sum().plot.bar(stacked=True,figsize=(11,6))
plt.xticks(rotation=70,fontsize=15)
plt.legend(loc=0, prop={'size': 14})
plt.title('Top 10 movies with most facebook likes',fontsize=18)
plt.show()


content=sorted_fl["content_rating"].head(200)
count=pd.Series(' '.join(content).lower().split(" ")).value_counts()[:5]
print("Most common ratings of top 150 movies with most facebook likes are:\n",count.to_frame('Count'))
dataset.loc[ dataset['imdb_score'] < 7.0, 'imdb_score'] = 0
dataset.loc[ dataset['imdb_score'] >= 7.0, 'imdb_score'] = 1

#Gradient Boosting
x=dataset[['movie_facebook_likes','title_year','budget','gross','director_facebook_likes','actors_facebook_likes','duration']]
y=dataset['imdb_score']
#train test split
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.4,random_state=13) 

n_trees=200
model=ensemble.GradientBoostingRegressor(loss='ls',learning_rate=0.03,n_estimators=n_trees,max_depth=4)
model.fit(x_train,y_train)

pred=model.predict(x_test)
error=model.loss_(y_test,pred) ##Loss function== Mean square error
print("MSE:%.3f" % error)

test_error=[]
for i,pred in enumerate(model.staged_predict(x_test)):##staged_predict=predict at each stage 
    test_error.append(model.loss_(y_test,pred))##model.loss(y_test,pred)=mse(y_test,pred)
    
plt.figure(figsize=(12,7))
plt.plot(list(range(1,n_trees+1)),model.train_score_,'b-',label='Train set error') ## model.train_score_=deviance(=loss) of model at each stage
plt.plot(list(range(1,n_trees+1)),test_error,'r-',label='Test set error')
plt.legend(loc='upper right',fontsize=15)
plt.xlabel('Trees',fontsize=15)
plt.ylabel('Error', fontsize=15)
plt.show()    

feature_importance=model.feature_importances_
sorted_importance = np.argsort(feature_importance)
pos=np.arange(len(sorted_importance))
plt.figure(figsize=(12,5))
plt.barh(pos, feature_importance[sorted_importance],align='center')
plt.yticks(pos, x.columns[sorted_importance],fontsize=15)
plt.title('Feature Importance ',fontsize=18)
plt.show()

#transformation to binary classification/regression
x=dataset[['movie_facebook_likes','title_year','budget','gross','director_facebook_likes','actors_facebook_likes','duration']]
y=dataset['imdb_score']
#train test split
#Logistic Regression
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.4,random_state=13)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
acc_log = round(logreg.score(x_train, y_train) * 100, 2)
acc_log
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)
#KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
acc_knn = round(knn.score(x_train, y_train) * 100, 2)
acc_knn
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)
#Random forest
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train, y_train)
y_pred = random_forest.predict(x_test)
random_forest.score(x_train, y_train)
acc_random_forest = round(random_forest.score(x_train, y_train) * 100, 2)
acc_random_forest
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)
#decision tree
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)
y_pred = decision_tree.predict(x_test)
acc_decision_tree = round(decision_tree.score(x_train, y_train) * 100, 2)
acc_decision_tree
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)
#
models = pd.DataFrame({
    'Model': [ 'KNN', 'Logistic Regression', 
              'Random Forest','Decision Tree'],
    'Score': [acc_knn, acc_log, acc_random_forest,acc_decision_tree]})
models.sort_values(by='Score', ascending=False)

dataset.describe('color')