#Sezione imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import confusion_matrix, auc, roc_curve, precision_score, recall_score, f1_score, accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from imblearn.under_sampling import TomekLinks
from sklearn.preprocessing import LabelBinarizer

dataset = pd.read_csv("Dataset/dermatologyReworked.dat")

print(dataset.shape)
print(dataset.info())
#print(dataset.describe())


#Esaminazione valori nulli per colonna

valori_nulli_per_colonna = dataset.isin(['?']).sum()
print(valori_nulli_per_colonna)

#Risoluzione valori nulli
moda_eta = dataset[" Age"].mode()
dataset[" Age"].replace("?", moda_eta[0], inplace=True)
dataset[" Age"] = dataset[" Age"].astype("int64")

#calcolo della matrice di correlazione
# Calcola la matrice di correlazione tra le features nel tuo DataFrame (supponiamo che il DataFrame si chiami 'dataset')
correlation_matrix = dataset.corr()

# Crea una heatmap utilizzando la matrice di correlazione
plt.figure(figsize=(20, 16))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap delle correlazioni tra le features')
plt.savefig("images/correlation_matrix.png")
plt.show()


#Esaminazione valori duplicati
valori_duplicati = dataset.duplicated()
duplicati_dataset = dataset[valori_duplicati]
print("\n\n Struttura del dataset dei duplicati")
print(duplicati_dataset.shape)

print("Struttura del dataset post drop")
dataset_nodup = dataset.drop_duplicates()
print(dataset_nodup.shape)


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
plt.savefig("images/plot_sbilanciamento.png")
plt.show()

#SEZIONE OVERSAMPLING

colonne = dataset.columns
colonne = colonne[:-1]

x = dataset[colonne]
y = dataset[" Class"]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=42)

#Oversampling con SMOTE
smote = SMOTE(random_state=42)
x_resampled, y_resampled = smote.fit_resample(x_train, y_train)

resampled_df = pd.DataFrame(data=x_resampled, columns=colonne)
resampled_df[" Class"] = y_resampled

plt.figure(figsize=(8,6))
sns.countplot(x=" Class", data=resampled_df, color='skyblue')
plt.title("SMOTE")
plt.xlabel("Malattia")
plt.ylabel("Conteggio")
plt.savefig("images/SMOTE_plot.png")
plt.show()

#Oversampling con Random Oversampling
ros = RandomOverSampler(random_state=42)
X_ros_resampled, Y_ros_resampled = ros.fit_resample(x_train, y_train)

ros_resampled_df = pd.DataFrame(data=X_ros_resampled, columns=colonne)
ros_resampled_df[" Class"] = Y_ros_resampled

plt.figure(figsize=(8, 6))
sns.countplot(x=" Class", data=ros_resampled_df)
plt.title("Random Oversampling")
plt.xlabel("Malattia")
plt.ylabel("Conteggio")
plt.savefig("images/random_oversampling_plot.png")
plt.show()


print("Dataset con SMOTE")
conteggio_classi_smote = resampled_df[" Class"].value_counts()
conteggio_classi_smote = conteggio_classi_smote.sort_index()
print(conteggio_classi_smote)

print("Dataset con Random Oversampling")
conteggio_classi_ros = ros_resampled_df[" Class"].value_counts()
conteggio_classi_ros = conteggio_classi_ros.sort_index()
print(conteggio_classi_ros)

#SEZIONE GESTIONE MODELLI
nb = BernoulliNB()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
knn = KNeighborsClassifier()
svc = SVC()


X = dataset[colonne]
y = dataset[" Class"]


X_smote = resampled_df[colonne]
y_smote = resampled_df[' Class']


X_ros = ros_resampled_df[colonne]
y_ros = ros_resampled_df[' Class']


#divisione dei dataset
#no oversampling
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#SMOTE
x_train_smote, x_test_smote, y_train_smote, y_test_smote = train_test_split(X_smote, y_smote, test_size=0.33, random_state=42)

#ROS
x_train_ros, x_test_ros, y_train_ros, y_test_ros = train_test_split(X_ros, y_ros, test_size=0.33, random_state=42)





#SEZIONE DECISIONAL TREE
#sezione con database senza oversampling

dt = dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)
conf_matrix = confusion_matrix(y_test, y_pred)

#sezione con smote

dt = dt.fit(x_train_smote, y_train_smote)
y_pred_smote = dt.predict(x_test_smote)
conf_matrix_smote = confusion_matrix(y_test_smote, y_pred_smote)

#sezione con random oversampling

dt = dt.fit(x_train_ros, y_train_ros)
y_pred_ros = dt.predict(x_test_ros)
conf_matrix_ros = confusion_matrix(y_test_ros, y_pred_ros)

#plot matrice di confusione
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='viridis', cbar=False)
plt.title("Matrice di Confusione senza oversampling")
plt.xlabel('Eticheta Predetta')
plt.ylabel('Eticheta Reale')
plt.savefig("images/dt_no_ovsp.png")
plt.show()

sns.heatmap(conf_matrix_smote, annot=True, fmt='d', cmap='viridis', cbar=False)
plt.title("Matrice di Confusione con SMOTE")
plt.xlabel('Eticheta Predetta')
plt.ylabel('Eticheta Reale')
plt.savefig("images/dt_smote.png")
plt.show()

sns.heatmap(conf_matrix_ros, annot=True, fmt='d', cmap='viridis', cbar=False)
plt.title("Matrice di Confusione con Random Oversampling")
plt.xlabel('Eticheta Predetta')
plt.ylabel('Eticheta Reale')
plt.savefig("images/dt_ros.png")
plt.show()

#SEZIONE PREDIZIONI
y_pred = dt.predict(x_test)
y_pred_smote = dt.predict(x_test_smote)
y_pred_ros = dt.predict(x_test_ros)

y_true = dataset[" Class"]

#CALCOLO DELLE PRESTAZIONI SENZA OVERSAMPLING
# Calcolo della precision
precision = precision_score(y_test, y_pred, average="macro")

# Calcolo della recall
recall = recall_score(y_test, y_pred, average="macro")

# Calcolo dell'F1 score
f1 = f1_score(y_test, y_pred, average="macro")

# Calcolo dell'accuracy
accuracy = accuracy_score(y_test, y_pred)

# Stampiamo i risultati
print("\nPRESTAZIONI DECISIONAL TREE")
print("Risultati senza oversampling")
print("Precision: %.4f" %precision)
print("Recall: %.4f" %recall)
print("F1 Score: %.4f" %f1)
print("Accuracy: %.4f" %accuracy)



#CALCOLO DELLE PRESTAZIONI CON SMOTE
precision_smote = precision_score(y_test_smote, y_pred_smote, average="macro")

# Calcolo della recall
recall_smote = recall_score(y_test_smote, y_pred_smote, average="macro")

# Calcolo dell'F1 score
f1_smote = f1_score(y_test_smote, y_pred_smote, average="macro")

# Calcolo dell'accuracy
accuracy_smote = accuracy_score(y_test_smote, y_pred_smote)

# Stampiamo i risultati
print("\nRisultati con SMOTE")
print("Precision: %.4f" %precision_smote)
print("Recall: %.4f" %recall_smote)
print("F1 Score: %.4f" %f1_smote)
print("Accuracy: %.4f" %accuracy_smote)


#CALCOLO DELLE PRESTAZIONI CON RANDOM OVERSAMPLING
precision_ros = precision_score(y_test_ros, y_pred_ros, average="macro")

# Calcolo della recall
recall_ros = recall_score(y_test_ros, y_pred_ros, average="macro")

# Calcolo dell'F1 score
f1_ros = f1_score(y_test_ros, y_pred_ros, average="macro")

# Calcolo dell'accuracy
accuracy_ros = accuracy_score(y_test_ros, y_pred_ros)

# Stampiamo i risultati
print("\nRisultati con Random Oversampling")
print("Precision: %.4f" %precision_ros)
print("Recall: %.4f" %recall_ros)
print("F1 Score: %.4f" %f1_ros)
print("Accuracy: %.4f " %accuracy_ros)


#SEZIONE NAIVE BAYES
#sezione con database senza oversampling
nb = dt.fit(x_train, y_train)
y_pred = nb.predict(x_test)
conf_matrix = confusion_matrix(y_test, y_pred)

#sezione con smote
nb = nb.fit(x_train_smote, y_train_smote)
y_pred_smote = nb.predict(x_test_smote)
conf_matrix_smote = confusion_matrix(y_test_smote, y_pred_smote)

#sezione con random oversampling
nb = nb.fit(x_train_ros, y_train_ros)
y_pred_ros = nb.predict(x_test_ros)
conf_matrix_ros = confusion_matrix(y_test_ros, y_pred_ros)

#plot matrice di confusione
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='viridis', cbar=False)
plt.title("Matrice di Confusione senza oversampling")
plt.xlabel('Eticheta Predetta')
plt.ylabel('Eticheta Reale')
plt.savefig("images/nb_no_ovsp.png")
plt.show()

sns.heatmap(conf_matrix_smote, annot=True, fmt='d', cmap='viridis', cbar=False)
plt.title("Matrice di Confusione con SMOTE")
plt.xlabel('Eticheta Predetta')
plt.ylabel('Eticheta Reale')
plt.savefig("images/nb_smote.png")
plt.show()

sns.heatmap(conf_matrix_ros, annot=True, fmt='d', cmap='viridis', cbar=False)
plt.title("Matrice di Confusione con Random Oversampling")
plt.xlabel('Eticheta Predetta')
plt.ylabel('Eticheta Reale')
plt.savefig("images/nb_ros.png")
plt.show()

#SEZIONE PREDIZIONI
y_pred = nb.predict(x_test)
y_pred_smote = nb.predict(x_test_smote)
y_pred_ros = nb.predict(x_test_ros)


#CALCOLO DELLE PRESTAZIONI SENZA OVERSAMPLING
# Calcolo della precision
precision = precision_score(y_test, y_pred, average="macro")

# Calcolo della recall
recall = recall_score(y_test, y_pred, average="macro")

# Calcolo dell'F1 score
f1 = f1_score(y_test, y_pred, average="macro")

# Calcolo dell'accuracy
accuracy = accuracy_score(y_test, y_pred)

# Stampiamo i risultati

print("\nPRESTAZIONI NAIVE BAYES")
print("Risultati senza oversampling")
print("Precision: %.4f" %precision)
print("Recall: %.4f" %recall)
print("F1 Score: %.4f" %f1)
print("Accuracy: %.4f" %accuracy)

#CALCOLO DELLE PRESTAZIONI CON SMOTE
precision_smote = precision_score(y_test_smote, y_pred_smote, average="macro")

# Calcolo della recall
recall_smote = recall_score(y_test_smote, y_pred_smote, average="macro")

# Calcolo dell'F1 score
f1_smote = f1_score(y_test_smote, y_pred_smote, average="macro")

# Calcolo dell'accuracy
accuracy_smote = accuracy_score(y_test_smote, y_pred_smote)

# Stampiamo i risultati
print("\nRisultati con SMOTE")
print("Precision: %.4f" %precision_smote)
print("Recall: %.4f" %recall_smote)
print("F1 Score: %.4f" %f1_smote)
print("Accuracy: %.4f" %accuracy_smote)


#CALCOLO DELLE PRESTAZIONI CON RANDOM OVERSAMPLING
precision_ros = precision_score(y_test_ros, y_pred_ros, average="macro")

# Calcolo della recall
recall_ros = recall_score(y_test_ros, y_pred_ros, average="macro")

# Calcolo dell'F1 score
f1_ros = f1_score(y_test_ros, y_pred_ros, average="macro")

# Calcolo dell'accuracy
accuracy_ros = accuracy_score(y_test_ros, y_pred_ros)

# Stampiamo i risultati
print("\nRisultati con Random Oversampling")
print("Precision: %.4f" %precision_ros)
print("Recall: %.4f" %recall_ros)
print("F1 Score: %.4f" %f1_ros)
print("Accuracy: %.4f " %accuracy_ros)


#RANDOM FOREST
#sezione con database senza oversampling
rf = rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
conf_matrix = confusion_matrix(y_test, y_pred)

#sezione con smote
rf = rf.fit(x_train_smote, y_train_smote)
y_pred_smote = rf.predict(x_test_smote)
conf_matrix_smote = confusion_matrix(y_test_smote, y_pred_smote)

#sezione con random oversampling
rf = rf.fit(x_train_ros, y_train_ros)
y_pred_ros = rf.predict(x_test_ros)
conf_matrix_ros = confusion_matrix(y_test_ros, y_pred_ros)

#plot matrice di confusione
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='viridis', cbar=False)
plt.title("Matrice di Confusione senza oversampling")
plt.xlabel('Eticheta Predetta')
plt.ylabel('Eticheta Reale')
plt.savefig("images/rf_no_ovsp.png")
plt.show()


sns.heatmap(conf_matrix_smote, annot=True, fmt='d', cmap='viridis', cbar=False)
plt.title("Matrice di Confusione con SMOTE")
plt.xlabel('Eticheta Predetta')
plt.ylabel('Eticheta Reale')
plt.savefig("images/rf_smote.png")
plt.show()

sns.heatmap(conf_matrix_ros, annot=True, fmt='d', cmap='viridis', cbar=False)
plt.title("Matrice di Confusione con Random Oversampling")
plt.xlabel('Eticheta Predetta')
plt.ylabel('Eticheta Reale')
plt.savefig("images/rf_ros.png")
plt.show()


#SEZIONE PREDIZIONI
y_pred = rf.predict(x_test)
y_pred_smote = rf.predict(x_test_smote)
y_pred_ros = rf.predict(x_test_ros)


#CALCOLO DELLE PRESTAZIONI SENZA OVERSAMPLING
# Calcolo della precision
precision = precision_score(y_test, y_pred, average="macro")

# Calcolo della recall
recall = recall_score(y_test, y_pred, average="macro")

# Calcolo dell'F1 score
f1 = f1_score(y_test, y_pred, average="macro")

# Calcolo dell'accuracy
accuracy = accuracy_score(y_test, y_pred)

# Stampiamo i risultati
print("\nPRESTAZIONI RANDOM FOREST")
print("Risultati senza oversampling")
print("Precision: %.4f" %precision)
print("Recall: %.4f" %recall)
print("F1 Score: %.4f" %f1)
print("Accuracy: %.4f" %accuracy)



#CALCOLO DELLE PRESTAZIONI CON SMOTE
precision_smote = precision_score(y_test_smote, y_pred_smote, average="macro")

# Calcolo della recall
recall_smote = recall_score(y_test_smote, y_pred_smote, average="macro")

# Calcolo dell'F1 score
f1_smote = f1_score(y_test_smote, y_pred_smote, average="macro")

# Calcolo dell'accuracy
accuracy_smote = accuracy_score(y_test_smote, y_pred_smote)

# Stampiamo i risultati
print("\nRisultati con SMOTE")
print("Precision: %.4f" %precision_smote)
print("Recall: %.4f" %recall_smote)
print("F1 Score: %.4f" %f1_smote)
print("Accuracy: %.4f" %accuracy_smote)


#CALCOLO DELLE PRESTAZIONI CON RANDOM OVERSAMPLING
precision_ros = precision_score(y_test_ros, y_pred_ros, average="macro")

# Calcolo della recall
recall_ros = recall_score(y_test_ros, y_pred_ros, average="macro")

# Calcolo dell'F1 score
f1_ros = f1_score(y_test_ros, y_pred_ros, average="macro")

# Calcolo dell'accuracy
accuracy_ros = accuracy_score(y_test_ros, y_pred_ros)

# Stampiamo i risultati
print("\nRisultati con Random Oversampling")
print("Precision: %.4f" %precision_ros)
print("Recall: %.4f" %recall_ros)
print("F1 Score: %.4f" %f1_ros)
print("Accuracy: %.4f " %accuracy_ros)



#SEZIONE KNN
#sezione con database senza oversampling
knn = knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
conf_matrix = confusion_matrix(y_test, y_pred)

#sezione con smote
knn = knn.fit(x_train_smote, y_train_smote)
y_pred_smote = knn.predict(x_test_smote)
conf_matrix_smote = confusion_matrix(y_test_smote, y_pred_smote)

#sezione con random oversampling
knn = knn.fit(x_train_ros, y_train_ros)
y_pred_ros = knn.predict(x_test_ros)
conf_matrix_ros = confusion_matrix(y_test_ros, y_pred_ros)

#plot matrice di confusione
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='viridis', cbar=False)
plt.title("Matrice di Confusione senza oversampling")
plt.xlabel('Eticheta Predetta')
plt.ylabel('Eticheta Reale')
plt.savefig("images/knn_no_ovsp.png")
plt.show()


sns.heatmap(conf_matrix_smote, annot=True, fmt='d', cmap='viridis', cbar=False)
plt.title("Matrice di Confusione con SMOTE")
plt.xlabel('Eticheta Predetta')
plt.ylabel('Eticheta Reale')
plt.savefig("images/knn_smote.png")
plt.show()

sns.heatmap(conf_matrix_ros, annot=True, fmt='d', cmap='viridis', cbar=False)
plt.title("Matrice di Confusione con Random Oversampling")
plt.xlabel('Eticheta Predetta')
plt.ylabel('Eticheta Reale')
plt.savefig("images/knn_ros.png")
plt.show()

#SEZIONE PREDIZIONI
y_pred = knn.predict(x_test)
y_pred_smote = knn.predict(x_test_smote)
y_pred_ros = knn.predict(x_test_ros)


#CALCOLO DELLE PRESTAZIONI SENZA OVERSAMPLING
# Calcolo della precision
precision = precision_score(y_test, y_pred, average="macro")

# Calcolo della recall
recall = recall_score(y_test, y_pred, average="macro")

# Calcolo dell'F1 score
f1 = f1_score(y_test, y_pred, average="macro")

# Calcolo dell'accuracy
accuracy = accuracy_score(y_test, y_pred)

# Stampiamo i risultati
print("\nPRESTAZIONI KNEAREST NEIGHBORS")
print("Risultati senza oversampling")
print("Precision: %.4f" %precision)
print("Recall: %.4f" %recall)
print("F1 Score: %.4f" %f1)
print("Accuracy: %.4f" %accuracy)



#CALCOLO DELLE PRESTAZIONI CON SMOTE
precision_smote = precision_score(y_test_smote, y_pred_smote, average="macro")

# Calcolo della recall
recall_smote = recall_score(y_test_smote, y_pred_smote, average="macro")

# Calcolo dell'F1 score
f1_smote = f1_score(y_test_smote, y_pred_smote, average="macro")

# Calcolo dell'accuracy
accuracy_smote = accuracy_score(y_test_smote, y_pred_smote)

# Stampiamo i risultati
print("\nRisultati con SMOTE")
print("Precision: %.4f" %precision_smote)
print("Recall: %.4f" %recall_smote)
print("F1 Score: %.4f" %f1_smote)
print("Accuracy: %.4f" %accuracy_smote)


#CALCOLO DELLE PRESTAZIONI CON RANDOM OVERSAMPLING
precision_ros = precision_score(y_test_ros, y_pred_ros, average="macro")

# Calcolo della recall
recall_ros = recall_score(y_test_ros, y_pred_ros, average="macro")

# Calcolo dell'F1 score
f1_ros = f1_score(y_test_ros, y_pred_ros, average="macro")

# Calcolo dell'accuracy
accuracy_ros = accuracy_score(y_test_ros, y_pred_ros)

# Stampiamo i risultati
print("\nRisultati con Random Oversampling")
print("Precision: %.4f" %precision_ros)
print("Recall: %.4f" %recall_ros)
print("F1 Score: %.4f" %f1_ros)
print("Accuracy: %.4f " %accuracy_ros)


#SEZIONE SVC

#sezione con database senza oversampling
svc = svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
conf_matrix = confusion_matrix(y_test, y_pred)

#sezione con smote
svc = svc.fit(x_train_smote, y_train_smote)
y_pred_smote = svc.predict(x_test_smote)
conf_matrix_smote = confusion_matrix(y_test_smote, y_pred_smote)

#sezione con random oversampling
svc = svc.fit(x_train_ros, y_train_ros)
y_pred_ros = svc.predict(x_test_ros)
conf_matrix_ros = confusion_matrix(y_test_ros, y_pred_ros)

#plot matrice di confusione
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='viridis', cbar=False)
plt.title("Matrice di Confusione senza oversampling")
plt.xlabel('Eticheta Predetta')
plt.ylabel('Eticheta Reale')
plt.savefig("images/svc_no_ovsp.png")
plt.show()

sns.heatmap(conf_matrix_smote, annot=True, fmt='d', cmap='viridis', cbar=False)
plt.title("Matrice di Confusione con SMOTE")
plt.xlabel('Eticheta Predetta')
plt.ylabel('Eticheta Reale')
plt.savefig("images/svc_smote.png")
plt.show()

sns.heatmap(conf_matrix_ros, annot=True, fmt='d', cmap='viridis', cbar=False)
plt.title("Matrice di Confusione con Random Oversampling")
plt.xlabel('Eticheta Predetta')
plt.ylabel('Eticheta Reale')
plt.savefig("images/svc_ros.png")
plt.show()

#SEZIONE PREDIZIONI
y_pred = svc.predict(x_test)
y_pred_smote = svc.predict(x_test_smote)
y_pred_ros = svc.predict(x_test_ros)


#CALCOLO DELLE PRESTAZIONI SENZA OVERSAMPLING
# Calcolo della precision
precision = precision_score(y_test, y_pred, average="macro")

# Calcolo della recall
recall = recall_score(y_test, y_pred, average="macro")

# Calcolo dell'F1 score
f1 = f1_score(y_test, y_pred, average="macro")

# Calcolo dell'accuracy
accuracy = accuracy_score(y_test, y_pred)

# Stampiamo i risultati
print("\nPRESTAZIONI SUPPORT VECTOR MACHINE")
print("Risultati senza oversampling")
print("Precision: %.4f" %precision)
print("Recall: %.4f" %recall)
print("F1 Score: %.4f" %f1)
print("Accuracy: %.4f" %accuracy)



#CALCOLO DELLE PRESTAZIONI CON SMOTE
precision_smote = precision_score(y_test_smote, y_pred_smote, average="macro")

# Calcolo della recall
recall_smote = recall_score(y_test_smote, y_pred_smote, average="macro")

# Calcolo dell'F1 score
f1_smote = f1_score(y_test_smote, y_pred_smote, average="macro")

# Calcolo dell'accuracy
accuracy_smote = accuracy_score(y_test_smote, y_pred_smote)

# Stampiamo i risultati
print("\nRisultati con SMOTE")
print("Precision: %.4f" %precision_smote)
print("Recall: %.4f" %recall_smote)
print("F1 Score: %.4f" %f1_smote)
print("Accuracy: %.4f" %accuracy_smote)


#CALCOLO DELLE PRESTAZIONI CON RANDOM OVERSAMPLING
precision_ros = precision_score(y_test_ros, y_pred_ros, average="macro")

# Calcolo della recall
recall_ros = recall_score(y_test_ros, y_pred_ros, average="macro")

# Calcolo dell'F1 score
f1_ros = f1_score(y_test_ros, y_pred_ros, average="macro")

# Calcolo dell'accuracy
accuracy_ros = accuracy_score(y_test_ros, y_pred_ros)

# Stampiamo i risultati
print("\nRisultati con Random Oversampling")
print("Precision: %.4f" %precision_ros)
print("Recall: %.4f" %recall_ros)
print("F1 Score: %.4f" %f1_ros)
print("Accuracy: %.4f " %accuracy_ros)
















