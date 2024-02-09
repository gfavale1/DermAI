import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import confusion_matrix, auc, roc_curve
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from imblearn.under_sampling import TomekLinks
from sklearn.preprocessing import LabelBinarizer


dataset = pd.read_csv("Dataset/dermatologyReworked.dat")

print(dataset.info())


#Esaminazione valori nulli per colonna

valori_nulli_per_colonna = dataset.isin(['?']).sum()
print(valori_nulli_per_colonna)

#Risoluzione valori nulli
moda_eta = dataset[" Age"].mode()
dataset[" Age"].replace("?",moda_eta[0],inplace=True)
dataset[" Age"] = dataset[" Age"].astype("int64")


#Esaminazione valori duplicati
valori_duplicati = dataset.duplicated()
duplicati_dataset = dataset[valori_duplicati]
print("\n\n")
print(duplicati_dataset)


# #Esaminazione delle distribuzioni dei dati
# for column in dataset.columns:
#     plt.figure(figsize=(6, 6))
#     data_min = dataset[column].min()
#     data_max = dataset[column].max()
#     step = 1
#     plt.hist(dataset[column], bins=20, color='skyblue', edgecolor='black', )# Modifica il numero di bins a tuo piacimento
#     plt.title(f'Distribuzione di {column}')
#     plt.xlabel('Valore')
#     plt.ylabel('Frequenza')
#     plt.grid(True)
#     plt.show()


#Esaminazione del bilanciamento delle classi
conteggio_classi = dataset[" Class"].value_counts()
conteggio_classi = conteggio_classi.sort_index()
print(conteggio_classi)

#grafico sbilanciamento delle classi
plt.figure(figsize=(8,6))
sns.countplot(x=' Class', data=dataset, color='skyblue')
plt.title("Sbilanciamento delle classi")
plt.xlabel("Malattia")
plt.ylabel("Conteggio")

plt.show()