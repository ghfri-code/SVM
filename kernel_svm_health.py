# Import Modules: 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd

#--------Define Functions:------------------ 
# Define Functions: 

# Plot Functions:

def plot_scores(scores, C_range, gamma_range, k, data):
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.get_cmap("Spectral_r"))
    plt.title(data + ' of kernel SVC (' + str(k) + '-time-' + str(k) + '-fold CV)')
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.show()
    
# Load & Prepare Dataset: 

def load_dataset(path):
    
    df = pd.read_table(path, delimiter=' ', header=None)
    
    X = df.drop(df.columns[13], axis=1)
    Y = np.array(df.drop(df.columns[[0,1,2,3,4,5,6,7,8,9,10,11,12]], axis=1))[:, 0]
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, Y

# Configure Parameters:

data_path = "./data/Health.dat"
Kernel = ['rbf'] 
C_range = np.array([0.01, 0.04, 0.1, 0.4, 1, 4, 10, 40])
gamma_range = 10. ** np.arange(-4, 4)
param_grid = dict(kernel=Kernel, gamma=gamma_range, C=C_range)

K = [5, 10]

# Train & Test: 

X, Y = load_dataset(data_path)

for k in K: # k-time-k-fold CV
    
    print('******************************************')
    print('')
    print(str(k) + '-time-' + str(k) + '-fold CV:')
    
    grid = GridSearchCV(SVC(), param_grid=param_grid,
                        cv=RepeatedStratifiedKFold(n_splits=k, n_repeats=k),
                        return_train_score=True)
    grid.fit(X, Y)
    
    print("The best classifier is: ", grid.best_estimator_)
    
    cv_results = grid.cv_results_
    
    train_acc = cv_results['mean_train_score']
    train_acc = np.array(train_acc).reshape(len(C_range), len(gamma_range))
    
    test_acc = cv_results['mean_test_score']
    test_acc = np.array(test_acc).reshape(len(C_range), len(gamma_range))
    
    train_std = cv_results['std_train_score']
    train_std = np.array(train_std).reshape(len(C_range), len(gamma_range))
    
    test_std = cv_results['std_test_score']
    test_std = np.array(test_std).reshape(len(C_range), len(gamma_range))
    
    print('Train Accuracy for best estimator')
    print(np.max(train_acc)*100)
    
    print('Test Accuracy for best estimator')
    print(np.max(test_acc)*100) # or print(grid.best_score_)

    # Plot Train & Test Scores:
    
    plot_scores(train_acc, C_range, gamma_range, k, 'train acc')
    plot_scores(test_acc, C_range, gamma_range, k, 'test acc')
    
    plot_scores(train_std, C_range, gamma_range, k, 'train std')
    plot_scores(test_std, C_range, gamma_range, k, 'test std')
    
