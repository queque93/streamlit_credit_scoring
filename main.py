import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np 


st.write('''# Application de visualisation des données et de prediction de credit
''')

st.sidebar.header("Les parametres d'entrées")
 
def user_input():
    Credit_History=st.sidebar.radio(
        "historique",
        ('Oui','Non',))
    Loan_Amount_Term = st.sidebar.number_input('Inserer une durée de prêt')
    CoapplicantIncome = st.sidebar.number_input("Inserer le revenu de l'aval")
    if Credit_History.lower()=='oui':
        Credit_History=1
    else:
        Credit_History=0
    data = {
        'Credit_History':Credit_History,
        'Loan_Amount_Term':Loan_Amount_Term,
        'CoapplicantIncome':CoapplicantIncome
    }
    #input_parametres=pd.DataFrame(data, index=[0])
    return data 

df = user_input()



dataset = pd.read_csv('CREDIT_SCORING.csv')
st.write("      ")
st.write("      ")
#voir la documentation pour separer
st.write(dataset)


st.subheader("Repartition selon le genre")

plt.figure(figsize=(12,4))
sns.factorplot('Gender',data=dataset,kind='count')
st.set_option('deprecation.showPyplotGlobalUse',False)
st.pyplot()

st.subheader("Repartition de l'accord de credit selon le genre")

fig, ax = plt.subplots(figsize=(12,8))

sns.countplot(data=dataset, x='Loan_Status',hue='Gender',ax=ax)
st.pyplot()

st.subheader("Repartition de l'accord de credit selon l'historique de credit")
#credit history
grid=sns.FacetGrid(dataset,col ='Loan_Status',size=4,aspect=1.6)
grid.map(sns.countplot,'Credit_History')
st.pyplot()

st.subheader("Repartition de l'accord de credit selon le statut matrimonial")
#statut matrimonial
grid=sns.FacetGrid(dataset,col ='Loan_Status',size=4,aspect=1.6)
grid.map(sns.countplot,'Married')
st.pyplot()

st.subheader("Repartition de montant de credit")
plt.hist(dataset['LoanAmount'])
st.pyplot()



st.subheader("Repartition de l'accord de credit selon le niveau d'education")
#niveau d'instruction
grid=sns.FacetGrid(dataset,col ='Loan_Status',size=4,aspect=1.6)
grid.map(sns.countplot,'Education')
st.pyplot()

st.subheader("Repartition de l'accord de credit selon le revenu et le genre")
sns.catplot(x="Loan_Status",y="ApplicantIncome",hue='Gender',kind="box",data=dataset)
st.pyplot()




def history_groups(series):
    if series <1:
        return "No"
    elif 1<= series :
        return "Yes"

dataset['Credit_History'] = dataset['Credit_History'].apply(history_groups)

#dataset['Credit_History'].value_counts(sort=False)

X = dataset.iloc[:,1:12].values
y = dataset.iloc[:,12].values

for i in list(dataset.columns)[1:-1]:
    #imputation des variables qualitatives 
    if dataset[i].dtype =='object':
        dataset[i].fillna(value=dataset[i].mode()[0],inplace=True)
    else:
    #imputation des variables quantitatives
        dataset[i].fillna(value=dataset[i].mean(),inplace=True) 

cat_data=[]
num_data=[]
for i,c in enumerate(dataset.dtypes):
    if c==object:
        cat_data.append(dataset.iloc[:,i])
    else:
        num_data.append(dataset.iloc[:,i])
cat_data=pd.DataFrame(cat_data).transpose()
num_data=pd.DataFrame(num_data).transpose()


#codage de la variable loan_status
target_value={'Y':1,'N':0}
target=cat_data['Loan_Status']
cat_data.drop('Loan_Status',axis=1,inplace=True)
target=target.map(target_value)

# methode 2 utilisation de la librairie scikit label encoder
le=LabelEncoder()
for i in cat_data:
    cat_data[i]=le.fit_transform(cat_data[i])
#cat_data
#supprimer Loans_id
cat_data.drop('Loan_ID',axis=1,inplace=True)
#concatener cat_data et num_data et specicfication de la variable cible
X=pd.concat([cat_data,num_data],axis=1)
y=target
#X=dataset.iloc[:,:11]
#y=dataset.iloc[:,-1]

#modelisation
#Division du dataset en donnée d'entrainement et donnée test
sss=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train,test in sss.split(X,y):
    X_train,X_test=X.iloc[train],X.iloc[test]
    y_train,y_test=y.iloc[train],y.iloc[test]
print('X_train  taille:', X_train.shape)
print('X_test   taille:', X_test.shape)
print('y_train  taille:', y_train.shape)
print('y_test   taille:', y_test.shape)

#Appliquons le meilleur modele a notre jeu de donnée(regression logistique)

##choix des variables a inclure dans le modele

import statsmodels.api as sm

X_train=X_train[['Credit_History','Loan_Amount_Term','CoapplicantIncome']]
logit_model=sm.Logit(y_train,X_train)
result=logit_model.fit()
result.summary2()

X_test=X_test[['Credit_History','Loan_Amount_Term','CoapplicantIncome']]
logit_model=sm.Logit(y_test,X_test)
result=logit_model.fit()
print(result.summary2())

from sklearn import metrics
classifier=LogisticRegression(random_state=42)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))

X_2=X[['Credit_History','Loan_Amount_Term','CoapplicantIncome']]
classifier=LogisticRegression()
classifier.fit(X_2,y)

def accord_credit(model,Credit_History,Loan_Amount_Term,CoapplicantIncome ):
    x = np.array([Credit_History,Loan_Amount_Term,CoapplicantIncome]).reshape(1,3)
    prediction = model.predict(x)
    if prediction==1:
        st.write("Credit accordé")
    else:
        st.write("Credit non accordé")
st.subheader("Prediction  d'accord credit")
accord_credit(classifier,Credit_History=df['Credit_History'],Loan_Amount_Term=df['Loan_Amount_Term'],CoapplicantIncome=df['CoapplicantIncome'])
