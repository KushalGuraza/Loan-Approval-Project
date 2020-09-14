
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, plot_confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import statistics
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import cross_val_score
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV


data = pd.read_csv("Z:\\courses\\GITHUB\\Loan Approval Project\\data.csv")

#summarising the data
data.head()
data.info()
des = data.describe()
print(des)
data.isnull().sum()

#checking the correlation
rcParams['figure.figsize'] = 15,7
sns.heatmap(data.corr(), annot = True)

# Nan slots
rcParams['figure.figsize'] = 10,5
sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap= 'viridis')

#no. of nan values missing out of total
msno.bar(data, figsize=(12,7))

#visualizes the correlation matrix about the locations of missing values in columns.
msno.heatmap(data, figsize=(12,7))

#Analysing and imputing missing values of gender
data["Gender"].fillna(data["Gender"].mode()[0], inplace = True)


#Analysing and imputing missing values of  married column
sns.set_style("whitegrid")
sns.countplot(x="Gender", hue="Married", data= data)

def fillmarried(cols):
    
    gender = cols[0]
    married = cols[1]
    
    if pd.isnull(married):
        
        if gender == "Male":
            return "Yes"
        
        else:
            return "No"
        
        
    else:
        return married
    
data['Married'] = data[['Gender','Married']].apply(fillmarried,axis=1)
        

#Analysing and imputing missing values of dependents
sns.countplot(x="Married",hue= "Dependents", data= data)

def filldependents(cols):
    
    married = cols[0]
    dependents = cols[1]
    
    if pd.isnull(dependents):
        
        if married == "No":
           return "0"
    
    else:
        return dependents
    
data["Dependents"] = data[["Married","Dependents"]].apply(filldependents, axis= 1)        
  
      
rcParams['figure.figsize'] = 15,7
sns.boxplot(x=data["ApplicantIncome"],y= data["Dependents"], hue= data["Married"], showmeans=True)
sns.boxplot(x=data["CoapplicantIncome"],y= data["Dependents"], hue= data["Married"], showmeans=True)
sns.boxplot(x= data["ApplicantIncome"] + data["CoapplicantIncome"],y= data["Dependents"], hue= data["Married"], showmeans=True)

#data["Dependents"].fillna("2", inplace = True) 
data["Dependents"].fillna("1", inplace = True) 


#Analysing and imputing missing values of Self_Employed Column
sns.countplot(x= data["Gender"], hue =data["Self_Employed"])
data["Self_Employed"].fillna(data["Self_Employed"].mode()[0], inplace = True)


#Analysing and imputing missing values of Loan Amount
sns.scatterplot(x= data["LoanAmount"], y= data["ApplicantIncome"], hue= data["Self_Employed"] )
sns.scatterplot(x= data["LoanAmount"], y= data["ApplicantIncome"], hue= data["Property_Area"] )
sns.boxplot(x=data["Property_Area"], y=data["LoanAmount"], showmeans = True )


#data["LoanAmount"].fillna(data["LoanAmount"].median(), inplace = True)

#finding the median for each Property area
Urban= []
Rural= []
SemiUrban= []

def Pa_La(cols):
    pa= cols[0]
    la= cols[1]
    
    if pa == "Urban":
        Urban.append(la)
        return "done"
    elif pa == "Rural":
        Rural.append(la)
        return "done"
    else: 
        SemiUrban.append(la)
        return "done"
    

data[["Property_Area","LoanAmount"]].apply(Pa_La, axis =1)
        

Urban = [i for i in Urban if pd.notnull(i)]
SemiUrban = [i for i in SemiUrban if pd.notnull(i)]
Rural = [i for i in Rural if pd.notnull(i)]


Median_Ur =statistics.median(Urban)
Median_Ru =statistics.median(Rural)
Median_SUr = statistics.median(SemiUrban)

#filling the median values
def fillloanamount(cols):
    
    pa = cols[0]
    la = cols[1]
    
    if pd.isnull(la):
        
        if pa == "Urban":
           return Median_Ur
        elif pa == "Semiurban":
           return Median_SUr
        else:
           return Median_Ru
       
    else:
          return la
    
data["LoanAmount"] = data[["Property_Area","LoanAmount"]].apply(fillloanamount, axis= 1) 



#Analysing and imputing missing values of Loan_Amount_Term column
sns.jointplot(x= data["LoanAmount"], y= data["Loan_Amount_Term"])

data["Loan_Amount_Term"].fillna(data["Loan_Amount_Term"].median(), inplace = True)

#Analysing and imputing missing values of credit History
sns.pairplot(data)
data["Credit_History"].fillna( data["Credit_History"].mode()[0] , inplace = True)


#checking the Nan Values
rcParams['figure.figsize'] = 10,5
sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap= 'viridis')



x = data.iloc[:,0:11]
y = data.iloc[:,-1]


#checking the if the dataset in balanced or imbalanced
sns.countplot(y)


#this is an imbalanced data set of yes:no = 1:0.5 , thus no need to use any sampling techniques
#ROS = RandomOverSampler()
#x_ros,y_ros = ROS.fit_sample(x,y)

#the data set is balanced now
#sns.countplot(y_ros)

x.info()

#dummy coding and scaling
x_dummy = x.select_dtypes(exclude= "number") 
x_dummy

columns_to_drop = x_dummy.columns

x_scaling = x.drop(columns_to_drop, axis = 1)
x_scaling

columns_scaling = x_scaling.columns

loan_id = pd.DataFrame(x_dummy.iloc[:,0])

x_dummy = pd.get_dummies(x_dummy.iloc[:,1:], drop_first = True)

X = pd.concat((x_scaling , x_dummy), axis = 1)

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state=0)

scaling = StandardScaler()
x_train = scaling.fit_transform(x_train) 
x_test =  scaling.transform(x_test)


#1st model
#Logistic Regression
LR = LogisticRegression()
LR.get_params().keys()
lr_params = [{'solver' : ['newton-cg', 'lbfgs','sag'], 'penalty' : ['l2'], 'C' : [100, 10, 1.0, 0.1, 0.01 ], 'fit_intercept': [True,False]},
             {'solver' : ['saga'],  'penalty' : ['elasticnet'] ,'C' : [100, 10, 1.0, 0.1, 0.01 ], 'fit_intercept': [True,False],'l1_ratio':np.linspace(0,1,10) },
             {'solver' : ['liblinear'] , 'penalty' : ['l1','l2'],'C' : [100, 10, 1.0, 0.1, 0.01], 'fit_intercept': [True,False] , 'intercept_scaling' : [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]} ]
             

rndcv_LR = RandomizedSearchCV(estimator = LR, param_distributions = lr_params, cv = 10, n_iter = 40, scoring= "accuracy" )
rndcv_LR.fit(x_train,y_train)
rndcv_LR.best_params_
LR_mean_accuracy=rndcv_LR.best_score_

LR = LogisticRegression (solver = "liblinear",penalty = 'l1',intercept_scaling = 0.7, fit_intercept = False, C= 1)
LR.fit(x_train,y_train)
y_pred = LR.predict(x_test)
LR_accuracy = accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))
plot_confusion_matrix(LR,x_test, y_test)


#2nd model
#KNN
KNN = KNeighborsClassifier()

knn_params= [{'algorithm': ['auto'],'n_neighbors':np.arange(start=1,stop=100,step=1),"weights":['uniform','distance'] }, 
             {'algorithm': ['ball_tree'],'n_neighbors':np.arange(start=1,stop=100,step=1),"weights":['uniform', 'distance'],"leaf_size":np.arange(start=1,stop=100,step=1)},
             {'algorithm': ['kd_tree'],'n_neighbors':np.arange(start=1,stop=100,step=1),"weights":['uniform', 'distance'],"leaf_size":np.arange(start=1,stop=100,step=1)},
             {'algorithm': ['brute'],'n_neighbors':np.arange(start=1,stop=100,step=1),"weights":['uniform', 'distance'] }]

rndcv_KNN = RandomizedSearchCV(estimator = KNN, param_distributions = knn_params, cv = 10, n_iter = 40, scoring= 'accuracy')
rndcv_KNN.fit(x_train,y_train)
knn_mean_accuracy =rndcv_KNN.best_score_
rndcv_KNN.best_params_


KNN = KNeighborsClassifier(weights = 'distance',n_neighbors= 22,leaf_size= 69,algorithm= 'kd_tree')
KNN.fit(x_train,y_train)
y_pred = KNN.predict(x_test)
KNN_accuracy = accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))
plot_confusion_matrix(KNN,x_test, y_test)


#3rd model
#RandomForest
randomforest = {"n_estimators" : [10,20,30,40,50,60,70,80,90,100,150,200,300,500,1000],"max_depth" : randint(1,5),"min_samples_split" : [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],"min_samples_leaf" : randint(1,5),"criterion" : ["gini","entropy"],"min_weight_fraction_leaf" : [0,0.1,0.2,0.3,0.4,0.5],"max_features" : randint(1,12),'bootstrap':[True,False]}

RF = RandomForestClassifier()

rndcv = RandomizedSearchCV(estimator = RF, param_distributions = randomforest, cv= 10, n_iter = 40 , scoring= "accuracy" )
rndcv.fit(x_train,y_train)
rndcv.best_params_
RF_mean_accuracy = rndcv.best_score_

RF = RandomForestClassifier(bootstrap= False, criterion= 'entropy', max_depth= 3, max_features= 5, min_samples_leaf= 4, min_samples_split= 0.7, min_weight_fraction_leaf= 0.1,n_estimators= 10)
RF.fit(x_train,y_train)
y_pred = RF.predict(x_test)
RF_accuracy = accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))
plot_confusion_matrix(RF,x_test, y_test)



#4th model
#Naive Bayes

NB = GaussianNB()
NB.fit(x_train,y_train)
y_pred = NB.predict(x_test)
NB_accuracy = accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))
plot_confusion_matrix(NB,x_test, y_test)

#5th model
#Support Vector Machine
svc = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
              {'C': [1, 10, 100, 1000], 'kernel': ['poly'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]


SVC = svm.SVC()

gdsrch = GridSearchCV(estimator = SVC,param_grid = svc,scoring = 'accuracy',cv = 10)
gdsrch.fit(x_train,y_train)
gdsrch.best_params_
svc_mean_accuracy = gdsrch.best_score_

SVC = svm.SVC(C= 1, kernel= 'linear')
SVC.fit(x_train,y_train)
y_pred = SVC.predict(x_test)
svc_accuracy =accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))
plot_confusion_matrix(SVC,x_test, y_test)


# achieved accuracy for testing data

print("For Logistic Regression =" + str(LR_accuracy*100))
print("For KNearest Neighbors =" + str(KNN_accuracy*100))
print("For Random Forest =" + str(RF_accuracy*100))
print("For Naive Bayes =" + str(NB_accuracy*100))
print("For Support Vector Machine =" + str(svc_accuracy*100))































