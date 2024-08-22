# Support Vector Machine (SVM)

## Linear SVM
Use **“Dataset1.mat”** which is a 2D and 2-class dataset to do this part and Train the SVM using two different values of the penalty parameter, i.e., C=1 and C=100.

### The data and the decision boundary Plot
![decision boundary1](https://github.com/Ghafarian-code/SVM/blob/master/images/dataset1/Figure_2.png)
![decision boundary100](https://github.com/Ghafarian-code/SVM/blob/master/images/dataset1/Figure_4.png)


## Kernel SVM for two-class problem 
In general, SVM is a linear classifier. When data are not linearly separable, Kernel SVM can be used. Here, we utilize SVM with RBF kernel for non-linear classification. Perform this part on **“Dataset2.mat”** and **“Health.dat”** datasets, so we train SVM with the penalty parameter C and the standard deviation for RBF kernel.
Note: For selecting the best model is better to test the values in multiplicative steps such as 0.01, 0.04,0.1, 0.4, 1, 4, 10 and 40. Therefore, 64 different models should be evaluated to select the best model.

### The data and the decision boundary plot for “Dataset2.mat” (for best model)
![decision boundary](https://github.com/Ghafarian-code/SVM/blob/master/images/dataset2/Figure_10.png)


## Kernel SVM for multi-class problem
SVM can be extended for multi-class classification. For this, two approaches are possible: one-against-one and one-against-all(one-against-rest). we apply one-against-all approach for **“Vehicle.dat”**
### The test accuracies of five- time-five-fold cross validation for different value of C and 𝜎 .
![test accuracies](https://github.com/Ghafarian-code/SVM/blob/master/images/vehicle/Figure_2.png)
### The test corresponding variances of five- time-five-fold cross validation for different value of C and 𝜎 .
![test variances](https://github.com/Ghafarian-code/SVM/blob/master/images/vehicle/Figure_4.png)
