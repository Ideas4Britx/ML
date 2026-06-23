from sklearn.datasets import fetch_openml
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,precision_recall_curve
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import calibration_curve #ver si modelo hace falta calibracion o no


mnist = fetch_openml('mnist_784',as_frame=False,cache=True) #al ser imagenes, queremos verlo en numpy no en panads

X,y = mnist.data, mnist.target

print(X.shape) #cada imagen 28x28

def plot_digit(image_data):
    image = image_data.reshape(28,28)
    plt.imshow(image,cmap='binary')
    plt.axis('off')

#some_digit=X[0]
#plot_digit(some_digit)
#plt.show()

X_train,X_test,y_train,y_test = X[:60000], X[60000:],y[:60000],y[60000:]

y_train_5 = (y_train=='5') #true for all 5s, False for other digits

print(y_train_5) #cada parte dle array 1d dice true o false, en caso que sea 5 o no
print(type(y_train_5))

#if we have skewed datasets for clasification is not the best way to train it and check only the accuracy

sgd_clf = SGDClassifier(random_state=42)

y_train_pred = cross_val_predict(sgd_clf,X_train,y_train_5,cv=3)#cv indica los folds con los que trabajaremos

cm = confusion_matrix(y_train_5,y_train_pred)
print(cm) #en este caso solo tengo una matriz, pero supongo que sera un 3d para mas nms. esta es primera fila negative,segunda pos
#primera columna son no 5sy sgund columna son 5s. primera fila son no 5s segunda 5s. 

cheat_cm = np.array([['true negatives','false positives'],['false negatives','true positives']])
print(cheat_cm)

#Precisión = 90 / (90 + 10) = 90% denemoinador penaliza efecividad
precision=precision_score(y_train_5,y_train_pred)
print(precision) #precision del modelo clasificando 5s

# recall,La pregunta que responde es — de todos los 5s reales que había, ¿cuántos encontré?
recall = recall_score(y_train_5,y_train_pred)
print(recall)

#f1 socre
f1= f1_score(y_train_5,y_train)
print(f1)

#treshold
y_scores = cross_val_predict(sgd_clf,X_train,y_train_5,cv=3,method='decision_function')
precisions,recalls,tresholds = precision_recall_curve(y_train_5,y_scores)

#iporatten treshold y ROC aunque no visto, tambien imporatnte calibracion



