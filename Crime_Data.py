
# coding: utf-8

# # Homework Project : Crime Prediction

# Basic import statements

# In[1]:

from sklearn import datasets, preprocessing
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn import tree,svm, linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.neural_network import MLPClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from IPython.display import display, Math, Latex


# ## Data Analysis

# 1) Create a new column called "highCrime" which has a boolean value based on "ViolentCrimesPerPop".
# 
# 2) Get the positive and negative percentage of the data.
# 
# 3) Remove communityname and fold as it does not have predictive power.

# In[2]:

data = pd.read_csv('/Users/Naveen/Downloads/crime/Crime Prediction Data/communities-crime-clean.csv')


# In[3]:

data['highCrime'] = data['ViolentCrimesPerPop'] > 0.1


# In[4]:

data.describe()


# In[5]:

data.head()


# ### What are the percentage of positive and negative instances in the dataset?

# In[6]:

total_rows = data['highCrime'].count()
print("Total number of rows : ", total_rows)
positive_rows = sum(data['highCrime'])
negative_rows = sum(data['highCrime']==False)
print("Negative rows : ",negative_rows)
print("Positive rows : ",positive_rows)
positive_percent = (positive_rows / total_rows) * 100
print("The percentage of positive instances : %.3f"%positive_percent)
negative_percent = (negative_rows / total_rows) * 100
print("The percentage of negative instances : %.3f"%negative_percent)


# In[7]:

data.pop('ViolentCrimesPerPop')
data.pop('fold')
data.pop('communityname')
label = data.pop('highCrime')


# # Decision Tree

# Decision tree is a modelling technique that gives the best features based on a specific criteria. "Gini" is used as a criteria to train the model and test using the test data.
# 
# 1) Gives top 10 important features.
# 
# 2) Outputs Normal accuracy, precision and recall as well as Cross validation accuracy, precision and recall.

# In[8]:

dt = tree.DecisionTreeClassifier()
dt.fit(data, label)
prediction = dt.predict(data)


# In[9]:

dt_accuracy_score = metrics.accuracy_score(label,prediction)
print("Training accuracy: ",dt_accuracy_score)
dt_precision = metrics.precision_score(label, prediction)
print("Training Precision : ",dt_precision)
dt_recall = metrics.recall_score(label, prediction)
print("Training Recall : ",dt_recall)


# In[10]:

imp = DataFrame(dt.feature_importances_, columns = ["Important"], index = data.columns).sort_values(['Important'], ascending = False)
#Top 10 features from decision tree classifier
imp.head(10)


# In[11]:

dt = tree.DecisionTreeClassifier(criterion = 'entropy')
dt.fit(data, label)
prediction = dt.predict(data)


# ### What are the training accuracy, precision, and recall for this tree? 

# In[12]:

dt_accuracy_score = metrics.accuracy_score(label,prediction)
print("Training accuracy: ",dt_accuracy_score)
dt_precision = metrics.precision_score(label, prediction)
print("Training Precision : ",dt_precision)
dt_recall = metrics.recall_score(label, prediction)
print("Training Recall : ",dt_recall)


# In[13]:

imp = DataFrame(dt.feature_importances_, columns = ["Important"], index = data.columns).sort_values(['Important'], ascending = False)
#Top 10 features from decision tree classifier
imp.head(10)


# ### What are the main features used for classification? Can you explain why they make sense (or not)?

# Trained decision tree with "Gini" and "Entropy" as a criterion and found that the main features are 
# 
# 1. PctKids2Par
# 
# 2. racePctWhite
# 
# 3. racePctHisp
# 
# 
# It does make sense, as the percentage of kids with two parents can lead to lack of parental attention. Where there are more percantage of kids, then there are fairly high chances of crime. Percentage of population of Hispanic and Whites has importance of 0.06 which signifies there will be very low crime rate in these communities.

# ### What are the 10-fold cross-validation accuracy, precision, and recall?

# In[15]:

cross = cross_val_score(dt,data, label, cv = 10, scoring = "accuracy")
print("Decision Tree CV Accuracy : %.3f"% cross.mean())
pres = cross_val_score(dt,data, label, cv = 10, scoring = "precision")
print("Decision Tree CV Precision : %.3f"% pres.mean())
rec = cross_val_score(dt,data, label, cv = 10, scoring = "recall")
print("Decision Tree CV Recall : %.3f"% rec.mean())


# ### Why are they different from the results in the previous test?

# In the previous test, the whole dataset was used for training the model and tested on the same dataset. This is called overfitting. In 10 fold cross validation, the data is divided into 10 subsets and one of the subset is used as test data while other subsets are used to train the model and this process is  repeated 10 times. Since the model is tested with the data which is not trained, the original accuracy of the model evaluated.

# # GAUSSIAN NB

# In[16]:

NB = GaussianNB()
NB.fit(data, label)
NB_pred = NB.predict(data)


# ### What is the 10-fold cross-validation accuracy, precision, and recall for this method? 

# In[17]:

NB_Accuracy = cross_val_score(NB,data,label, cv =10, scoring = "accuracy")
print("Naive bayes accuracy : %.3f"%NB_Accuracy.mean())
NB_pres = cross_val_score(NB,data, label, cv = 10, scoring = "precision")
print("Naive bayes precision :%.3f"% NB_pres.mean())
NB_rec = cross_val_score(NB,data, label, cv = 10, scoring = "recall")
print("Naive bayes recall :%.3f"% NB_rec.mean())


# ### i.	What are the 10 most predictive features? This can be measured by the normalized absolute difference of means for the feature between the two classes:
# ### $$\frac{|\mu_T + \mu_F|} {\sigma_T + \sigma_F}$$
# ### The larger this different, the more predictive the feature. Why do these make sense (or not)?

# In[18]:

#Mean of each feature per class
NB_means = NB.theta_
NB_means_true = NB_means[[0]].ravel()
NB_means_false = NB_means[[1]].ravel()
#Variance of each feature per class
NB_sigma = NB.sigma_
NB_sigma_true = NB_sigma[[0]].ravel()
NB_sigma_false = NB_sigma[[1]].ravel()


# In[19]:

NB_feature = []

for i in list(range(len(NB_means_true))):
    numerator = np.absolute(NB_means_true[i] - NB_means_false[i])
    denominator = np.sqrt(NB_sigma_true[i]) + np.sqrt(NB_sigma_false[i])
    NB_feature.append(numerator / denominator)


# Printing the top 10 features

# In[20]:

best_NB = DataFrame(NB_feature, columns = ["Absolute"], index = data.columns).sort_values(['Absolute'], ascending = False)
best_NB.head(10)


# The most predictivee features makes sense because, kids with 2 parents and families with 2 parents will have more conflicts as no proper parental attention or guidance is given to them. Hence crimes can be high. Crimes can be more in the region where the female is divorced as there is no one in the house to support. Percentage of whites shows that if the number is more, crimes will be high.

# ### How do these results compare with your results from decision trees, above?
# 

# The top 2 features is the same as compared to decision tree. However, the feature importance values are more which can be used as a concrete support to our assumption. Also the accuracy of Gaussian Naive bayes is slightly higher than decision tree. 

# # LINEAR SVM

# In[21]:

lsvc = svm.LinearSVC()
lsvc.fit(data,label)
lsvc_pred = lsvc.predict(data)


# ### What is the 10-fold cross-validation accuracy, precision, and recall for this method?

# In[22]:

lsvc_accuracy = cross_val_score(lsvc, data,label, cv = 10, scoring = "accuracy")
print("Linear SVM Accuracy: %.3f"%lsvc_accuracy.mean())
lsvc_pres = cross_val_score(lsvc,data, label, cv = 10, scoring = "precision")
print("Linear SVM precision: %.3f"% lsvc_pres.mean())
lsvc_rec = cross_val_score(lsvc,data, label, cv = 10, scoring = "recall")
print("Linear SVM recall: %.3f"% lsvc_rec.mean())


# ### What are the 10 most predictive features? This can be measured by the absolute feature weights in the model. Why do these make sense (or not)? 

# In[23]:

best_lsvc = DataFrame(np.absolute(lsvc.coef_).ravel(), columns = ["Weights"], index = data.columns).sort_values(['Weights'], ascending = False)
best_lsvc.head(10)


# These features can also make sense, as the number of people living in the same city for 5 years means that they know the neighbours very well which can help in deciding the crime factor. Also the per capita income of native americans can help in deciding the crime rate.

# ### How do these results compare with your results from decision trees, above? 

# Population of whites are in common with the decision tree prediction. But the accuracy of decision is more when compared to linear SVM. Hence decision tree results are better.

# # REGRESSION

# In[24]:

clean_data = pd.read_csv('/Users/Naveen/Downloads/crime/Crime Prediction Data/communities-crime-clean.csv')


# In[25]:

clean_data.pop('fold')
clean_data.pop('communityname')
clean_target = clean_data.pop('ViolentCrimesPerPop')


# ### Using 10-fold cross-validation, what is the estimated mean-squared-error (MSE) of the model? 

# In[26]:

linear_regression = linear_model.LinearRegression()
reg_cross = np.absolute(cross_val_score(linear_regression, clean_data, clean_target,cv = 10, scoring = 'neg_mean_squared_error'))
print("Mean square error from cross validation : %.3f"%reg_cross.mean())


# ### What is the MSE on the training set (train on all the data then test on it all)? 

# In[27]:

linear_regression.fit(clean_data, clean_target)
print("Mean squared error: %.3f"% np.mean((linear_regression.predict(clean_data) - clean_target) ** 2))


# In[28]:

print("Accuracy : %.3f"%linear_regression.score(clean_data, clean_target))


# ### What features are most predictive of a high crime rate? A low crime rate?

# In[29]:

#Checking F values of features
fvalue = f_regression(clean_data, clean_target)[0]
best_reg = DataFrame(fvalue, columns = ["f value"], index = clean_data.columns).sort_values(['f value'], ascending = False)
best_reg.head(10)


# In[30]:

#Checking p value of F scores
p_value = f_regression(clean_data, clean_target)[1]
best_reg = DataFrame(p_value, columns = ["p value"], index = clean_data.columns).sort_values(['p value'], ascending = True)
best_reg.head(10)


# In[31]:

plt.plot(linear_regression.predict(clean_data), clean_target, 'o')
plt.show()


# # RIDGECV

# ### What is the estimated MSE of the model under 10-fold CV?

# In[32]:

ridge = linear_model.RidgeCV(alphas =(10,1,0.1,0.01,0.001) , store_cv_values= True)
print("MSE 10 fold : %.3f"%np.absolute(cross_val_score(ridge, clean_data, clean_target,cv = 10, scoring = 'neg_mean_squared_error').mean()))


# ### What is the MSE on the training set (train on all the data then test on it all)?

# In[33]:

ridge.fit(clean_data, clean_target)
print("MSE total: %.3f"%ridge.cv_values_.mean())


# ### What is the best alpha?

# In[34]:

print("The best alpha is:",ridge.alpha_)


# ### What does this say about the amount of overfitting in linear regression for this problem?

# In[35]:

ridge1 = linear_model.Ridge(alpha = 1.0)
ridge1.fit(clean_data, clean_target)


# In[36]:

plt.plot(ridge1.predict(clean_data), clean_target, 'o')
plt.show()


# The mean squared error for linear regression is almost the same as with ridge regression. Hence there is not much of the overfitting in linear regression.

# # Polynomial Features

# In[37]:

poly = PolynomialFeatures(degree = 2)
poly_transform = poly.fit_transform(clean_data)


# ### What is the estimated MSE of the model under 10-fold CV?

# In[38]:

print("Mean squared error : %.3f"%np.absolute((cross_val_score(linear_regression, poly_transform, clean_target, scoring = 'neg_mean_squared_error').mean())))


# ### What is the MSE on the training set (train on all the data then test on it all)?

# In[39]:

linear_regression.fit(poly_transform, clean_target)
print("Mean squared error: %.3f"% np.mean((linear_regression.predict(poly_transform) - clean_target) ** 2))


# In[40]:

plt.plot(linear_regression.predict(poly_transform), clean_target, 'o')
plt.show()


# ### Does this mean the quadratic model is better than the linear model for this problem?

# Yes. The quadratic model is better than the linear model as the mean squared error is 0.

# # DIRTY DATA Decision Tree

# Full dataset is used to train the decision tree and the missing values are filled with mean of their respective columns.

# In[41]:

full = pd.read_csv('/Users/Naveen/Downloads/crime/Crime Prediction Data/communities-crime-full.csv', na_values = ["?"])
full.describe()


# In[42]:

full = full.fillna(full.mean())


# In[43]:

full.head(10)


# In[44]:

full['highCrime'] = full['ViolentCrimesPerPop'] > 0.1


# In[45]:

total_rows = full['highCrime'].count()
print("Total number of rows : ", total_rows)
positive_rows = sum(full['highCrime'])
negative_rows = sum(full['highCrime']==False)
print("Negative rows : ",negative_rows)
print("Positive rows : ",positive_rows)
positive_percent = (positive_rows / total_rows) * 100
print("The percentage of positive instances : %.3f"%positive_percent)
negative_percent = (negative_rows / total_rows) * 100
print("The percentage of negative instances : %.3f"%negative_percent)


# In[46]:

full.pop('ViolentCrimesPerPop')
full.pop('communityname')
full.pop('fold')
label = full.pop('highCrime')


# In[47]:

dt = tree.DecisionTreeClassifier()
dt.fit(full, label)


# ### What are the training accuracy, precision, and recall for this tree?

# In[48]:

prediction = dt.predict(full)
accuracy_score_test = metrics.accuracy_score(label,prediction)
print("Training accuracy with full data : ",accuracy_score_test)
precision_test = metrics.precision_score(label, prediction)
print("Training Precision with full data : ",precision_test)
recall_test = metrics.recall_score(label, prediction)
print("Training Recall with full data : ",recall_test)


# In[49]:

imp = DataFrame(dt.feature_importances_, columns = ["Important"], index = full.columns).sort_values(['Important'], ascending = False)
imp.head(10)


# ### What are the 10-fold cross-validation accuracy, precision, and recall?

# In[50]:

cross = cross_val_score(dt,full, label, cv = 10, scoring = "accuracy")
print("Decision tree CV Accuracy with full data : %.3f"% cross.mean())
pres = cross_val_score(dt,full, label, cv = 10, scoring = "precision")
print("Decision tree CV Precision with full data : %.3f"% pres.mean())
rec = cross_val_score(dt,full, label, cv = 10, scoring = "recall")
print("Decision tree CV Recall with full data : %.3f"% rec.mean())


# ### Are the CV results better or worse? What does this say about the effect of missing values?

# When compared with the CV results of cleaned data, the CV results of full data is better and improved. However, the most important features seems to be the same. Hence for a model to learn and predict with high accuracy, missing values cannot be discarded and must be taken into consideration. In this case, 14 columns had missing values and all the 14 columns were removed. If there are too many columns with missing values, it is always a better practice to have a closer look on the data and consider filling those values with either zero, mean, median or mode.

# # Extra Stuff

# In[51]:

data = pd.read_csv('/Users/Naveen/Downloads/crime/Crime Prediction Data/communities-crime-clean.csv')


# In[52]:

data['highCrime'] = data['ViolentCrimesPerPop'] > 0.1


# In[53]:

data.pop('ViolentCrimesPerPop')
data.pop('fold')
data.pop('communityname')
label = data.pop('highCrime')


# In[54]:

full = pd.read_csv('/Users/Naveen/Downloads/crime/Crime Prediction Data/communities-crime-full.csv', na_values = ["?"])


# In[55]:

full = full.fillna(full.mean())


# In[56]:

full['highCrime'] = full['ViolentCrimesPerPop'] > 0.1


# In[57]:

full.pop('ViolentCrimesPerPop')
full.pop('fold')
full.pop('communityname')
target = full.pop('highCrime')


# # non linear SVC

# In[58]:

non_linear_svc = svm.SVC(kernel = 'rbf')


# In[59]:

non_linear_svc.fit(data, label)
non_linear_pred = non_linear_svc.predict(data)


# ### What are the 10-fold cross-validation accuracy, precision, and recall on cleaned dataset?

# In[60]:

nlsvc_accuracy = cross_val_score(non_linear_svc, data,label, cv = 10, scoring = "accuracy")
print("Non Linear Cleaned data SVC Accuracy: %.3f"%nlsvc_accuracy.mean())
nlsvc_pres = cross_val_score(non_linear_svc, data, label, cv = 10, scoring = "precision")
print("Non Linear Cleaned data SVC precision: %.3f"% nlsvc_pres.mean())
nlsvc_rec = cross_val_score(non_linear_svc, data, label, cv = 10, scoring = "recall")
print("Non Linear Cleaned data SVC recall: %.3f"% nlsvc_rec.mean())


# ### What are the 10-fold cross-validation accuracy, precision, and recall on full dataset?

# In[61]:

nlsvc_accuracy = cross_val_score(non_linear_svc, full,target, cv = 10, scoring = "accuracy")
print("Non Linear full data SVC Accuracy: %.3f"%nlsvc_accuracy.mean())
nlsvc_pres = cross_val_score(non_linear_svc, full, target, cv = 10, scoring = "precision")
print("Non Linear full data SVC precision: %.3f"% nlsvc_pres.mean())
nlsvc_rec = cross_val_score(non_linear_svc, full, target, cv = 10, scoring = "recall")
print("Non Linear full data SVC recall: %.3f"% nlsvc_rec.mean())


# # Random Forest

# In[62]:

random_forest = RandomForestClassifier(n_estimators = 50)


# In[63]:

random_forest.fit(data, label)


# In[64]:

random_forest_pred = random_forest.predict(data)


# ### What are the 10-fold cross-validation accuracy, precision, and recall on cleaned dataset?

# In[65]:

random_accuracy = cross_val_score(random_forest, data,label, cv = 10, scoring = "accuracy")
print("Random Forest Accuracy:",random_accuracy.mean())
random_precision = cross_val_score(random_forest,data, label, cv = 10, scoring = "precision")
print("Random Forest precision:" , random_precision.mean())
random_rec = cross_val_score(random_forest,data, label, cv = 10, scoring = "recall")
print("Random Forest recall:" , random_rec.mean())


# In[66]:

random_imp = DataFrame(random_forest.feature_importances_, columns = ["Imp"], index = data.columns).sort_values(['Imp'], ascending = False)
random_imp.head(10)


# In[67]:

conf_matrix = confusion_matrix(label, random_forest_pred)
conf_matrix


# ### What are the 10-fold cross-validation accuracy, precision, and recall on full dataset?

# In[68]:

random_accuracy = cross_val_score(random_forest, full,target, cv = 10, scoring = "accuracy")
print("Random Forest Accuracy:",random_accuracy.mean())
random_precision = cross_val_score(random_forest,full,target, cv = 10, scoring = "precision")
print("Random Forest precision:" , random_precision.mean())
random_rec = cross_val_score(random_forest,full,target, cv = 10, scoring = "recall")
print("Random Forest recall:" , random_rec.mean())


# In[69]:

random_forest.fit(full, target)
random_forest_pred = random_forest.predict(full)
random_imp = DataFrame(random_forest.feature_importances_, columns = ["Imp"], index = full.columns).sort_values(['Imp'], ascending = False)
random_imp.head(10)


# In[70]:

conf_matrix = confusion_matrix(target, random_forest_pred)
conf_matrix


# # Gradient Boosting

# In[71]:

grd = GradientBoostingClassifier()
grd.fit(data, label)
grd_pred = grd.predict(data)


# ### What are the 10-fold cross-validation accuracy, precision, and recall on cleaned dataset?

# In[72]:

grd_accuracy = cross_val_score(grd, data,label, cv = 10, scoring = "accuracy")
print("Gradient Boosting Cleaned data Accuracy:%.3f"%grd_accuracy.mean())
grd_pres = cross_val_score(grd,data, label, cv = 10, scoring = "precision")
print("Gradient Boosting Cleaned data precision:%.3f"% grd_pres.mean())
grd_rec = cross_val_score(grd,data, label, cv = 10, scoring = "recall")
print("Gradient Boosting Cleaned data recall:%.3f"% grd_rec.mean())


# In[73]:

grd_imp = DataFrame(grd.feature_importances_, columns = ["Imp"], index = data.columns).sort_values(['Imp'], ascending = False)
grd_imp.head(10)


# In[74]:

grd = GradientBoostingClassifier()
grd.fit(full, target)
grd_pred = grd.predict(full)


# ### What are the 10-fold cross-validation accuracy, precision, and recall on full dataset?

# In[75]:

grd_accuracy = cross_val_score(grd, full,target, cv = 10, scoring = "accuracy")
print("Gradient Boosting Full data Accuracy: %.3f"%grd_accuracy.mean())
grd_pres = cross_val_score(grd,full,target, cv = 10, scoring = "precision")
print("Gradient Boosting Full data precision:%.3f"% grd_pres.mean())
grd_rec = cross_val_score(grd,full,target, cv = 10, scoring = "recall")
print("Gradient Boosting Full data recall:%.3f"% grd_rec.mean())


# In[76]:

grd_imp = DataFrame(grd.feature_importances_, columns = ["Imp"], index = full.columns).sort_values(['Imp'], ascending = False)
grd_imp.head(10)


# # Neural Networks

# In[77]:

neural = MLPClassifier()
neural.fit(data,label)
neural_pred = neural.predict(data)


# ### What are the 10-fold cross-validation accuracy, precision, and recall on cleaned dataset?

# In[78]:

neural_accuracy = cross_val_score(neural, data,label, cv = 10, scoring = "accuracy")
print("Neural network Cleaned data Accuracy:",neural_accuracy.mean())
neural_pres = cross_val_score(neural,data, label, cv = 10, scoring = "precision")
print("Neural network Cleaned data precision:" , neural_pres.mean())
neural_rec = cross_val_score(neural,data, label, cv = 10, scoring = "recall")
print("Neural network Cleaned data recall:" , neural_rec.mean())


# In[79]:

neural.fit(full,target)
neural_pred = neural.predict(full)


# ### What are the 10-fold cross-validation accuracy, precision, and recall on full dataset?

# In[80]:

neural_accuracy = cross_val_score(neural, full,target, cv = 10, scoring = "accuracy")
print("Neural network Full data Accuracy:",neural_accuracy.mean())
neural_pres = cross_val_score(neural,full,target, cv = 10, scoring = "precision")
print("Neural network Full data precision:" , neural_pres.mean())
neural_rec = cross_val_score(neural,full,target, cv = 10, scoring = "recall")
print("Neural network Full data recall:" , neural_rec.mean())


# # Final conclusion

# ### What method gives the best results?

# Random Forest gives the best CV results for both cleaned and full dataset. Gradient boosting is equally as good as random forest, But when compared to the CV results of both cleaned and full dataset, Random Forest is better.

# ### What feature(s) seem to be most consistently predictive of high crime rates? How reliable is this conclusion?

# The most important and consistent features are listed below.
# 
# 1. Percentage of kids with 2 parents.
# 2. Population percentage of Whites.
# 
# kids with 2 parents and families with 2 parents will have more conflicts with no proper parental attention or guidance. Hence crimes can be high. The community with more Whites population have high crime rate. 
