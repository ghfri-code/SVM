# Support Vector Machine (SVM)

## Linear SVM
Use “Dataset1.mat” which is a 2D and 2-class dataset to do this part and Train the SVM using two different values of the penalty parameter, i.e., C=1band C=100.

### Plot the data and the decision boundary:

![alt text](https://github.com/Ghafarian-code/SVM/blob/master/images/dataset1/Figure_2.png)
![alt text](https://github.com/Ghafarian-code/SVM/blob/master/images/dataset1/Figure_4.png)


## Kernel SVM for two-class problem 
In general, SVM is a linear classifier. When data are not linearly separable, Kernel SVM can be used. Here, we utilize SVM with RBF kernel for non-linear classification. Perform the
following step for “Dataset2.mat” and “Health.dat” datasets, so we train SVM with the penalty parameter C and the standard deviation for RBF kernel. Determine the best value C by ten-time-ten-fold cross validation.
Note: It is better to test the values in multiplicative steps such as 0.01, 0.04,0.1, 0.4, 1, 4, 10 and 40. Therefore, you should evaluate 64 (82) different models to select the best model.

### Plot train and test accuracies and their corresponding variances of five-time- five-fold cross validation for different values of C and 𝜎 .
![alt text](https://github.com/Ghafarian-code/SVM/blob/master/images/dataset1/Figure_4.png)

### Plot the data and the decision boundary for “Dataset2.mat” (for best model)
![alt text](https://github.com/Ghafarian-code/SVM/blob/master/images/dataset1/Figure_4.png)

## Kernel SVM for multi-class problem
SVM can be extended for multi-class classification. For this, two approaches are possible: one-against-one and one-against-all(one-against-rest). Use one-against-all method (Dataset:Vehicle.dat)
### Plot the train and test accuracies and their corresponding variances of five- time-five-fold cross validation for different value of C and 𝜎 .
![alt text](https://github.com/Ghafarian-code/SVM/blob/master/images/dataset1/Figure_4.png)
