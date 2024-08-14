#--------Import Modules: -----------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import scipy.io

#--------Define Functions:------------------ 
# Plot Functions:

def make_meshgrid(X1, X2, h=.02):
    x_min, x_max = X1.min() - 1, X1.max() + 1
    y_min, y_max = X2.min() - 1, X2.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, svclassifier, xx, yy, **params):
    Z = svclassifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

# Load & Prepare Dataset: 
    
def load_dataset(path, test_split):

    X = scipy.io.loadmat(path)['X']
    Y = scipy.io.loadmat(path)['y'][:, 0]
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_split, random_state=0)
    
    return X_train, X_test, y_train, y_test


#---------Configure Parameters: ----------------
    
data_path = "./data/Dataset1.mat"
C = [1.0, 100.0]
Kernel = 'linear'

#---------Call functions ----------------------
# Train & Test: 

X_train, X_test, y_train, y_test = load_dataset(data_path, 0.25)

for param_C in C:

    svclassifier = SVC(kernel=Kernel, C=param_C)
    svclassifier.fit(X_train, y_train)
    
    y_pred_train = svclassifier.predict(X_train)
    y_pred_test = svclassifier.predict(X_test)

    
    print('******************************************')
    print('Train Accuracy for C = ' + str(param_C))
    print(accuracy_score(y_train, y_pred_train)*100)
    
    print('******************************************')
    print('Test Accuracy for C = ' + str(param_C))
    print(accuracy_score(y_test, y_pred_test)*100)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    plt.show()
        


    #plotting
    fig, ax = plt.subplots()
    # title for the plots
    title = ('Decision surface of linear SVC (C = ' + str(param_C) + ')')    
    X0, X1 = X_test[:, 0], X_test[:, 1]
    xx, yy = make_meshgrid(X0, X1) 
    plot_contours(ax, svclassifier, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8) 
    ax.scatter(X0, X1, c=y_pred_test, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    
    ax.set_ylabel('Feature 1')
    ax.set_xlabel('Feature 2')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    #ax.legend()
    plt.show()
    
