import numpy as np
from KNN import KNN
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
cmap = ListedColormap(['#FF0000' , '#00FF00' , '#0000FF'])
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay

iris = datasets.load_iris()
X , y = iris.data , iris.target

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=1234)

plt.figure()
plt.scatter(X[:,2],X[:,3] , c=y , cmap=cmap , edgecolor='k' , s=20)
plt.show()

clf=KNN(k=5)
clf.fit(X_train , y_train)
predictions = clf.predict(X_test)



acc = np.sum(predictions == y_test) / len(y_test)
print(acc)

cm = confusion_matrix(y_test , predictions)
cm_dis = ConfusionMatrixDisplay(cm)
cm_dis.plot()
plt.show()