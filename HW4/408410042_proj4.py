#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from libsvm.svmutil import *
from sklearn.preprocessing import MinMaxScaler


# In[2]:


# Define the linear model
def linear_model(x):
    return 2*x + np.random.normal(0, 1)

# Generate 500 data points with equal spacing in the range [-100, 100]
x = np.linspace(-100, 100, 500)
y = np.array([linear_model(xi) for xi in x])

# Plot the data
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('500 Data Points Generated with Linear Model y = 2x + ε')
plt.show()


# In[3]:


x = np.linspace(-100, 100, 500)
y = np.array([linear_model(xi) for xi in x])

# formatting it as required by LIBSVM
data = [{'label': int(y[i]), 'features': [x[i]]} for i in range(len(x))]

# 5-fold cross-validation procedure 
num_folds = 5
subset_size = len(data) // num_folds


# In[4]:


def split_data(data):
    np.random.shuffle(data)
    num_samples = len(data)
    num_train = int(num_samples * 0.8)
    train_data = data[:num_train]
    test_data = data[num_train:]
    return train_data, test_data


def scale_data(data, scaler):
    X = [d['features'] for d in data]
    X_scaled = scaler.fit_transform(X)
    scaled_data = []
    for i in range(len(data)):
        scaled_data.append({'label': data[i]['label'], 'features': X_scaled[i]})
    return scaled_data


# In[5]:


def train_and_evaluate_svm(data, c, gamma, scaling=True):
    # Split data into training and testing sets
    train_data, test_data = split_data(data)
    
    if scaling:
        # Scale the feature values to be between 0 and 1
        scaler = MinMaxScaler()
        train_data = scale_data(train_data, scaler)
        test_data = scale_data(test_data, scaler)
    
    best_model = None
    best_accuracy = 0
    
    # Train and evaluate SVM models for all combinations of C and gamma
   
    print('Training SVM with C = {}, gamma = {}'.format(c, gamma))

    # Write the training and testing data to files
    with open('train.dat', 'w') as f:
        for d in train_data:
            f.write('{} {}\n'.format(d['label'], ' '.join('{}:{}'.format(i+1, x) for i, x in enumerate(d['features']))))
    with open('test.dat', 'w') as f:
        for d in test_data:
            f.write('{} {}\n'.format(d['label'], ' '.join('{}:{}'.format(i+1, x) for i, x in enumerate(d['features']))))

    # Load the training and testing data from the files
    y_train, x_train = svm_read_problem('train.dat')
    y_test, x_test = svm_read_problem('test.dat')

    # Train the SVM model
    model = svm_train(y_train, x_train, '-s 0 -t 2 -c {} -g {}'.format(c, gamma))

    # Evaluate the SVM model on the testing data
    p_label, p_acc, p_val = svm_predict(y_test, x_test, model)
    
    return p_acc[0]


# In[16]:


best_params = []

# Define the parameter grids for C and gamma
C_values = [100000, 150000, 200000]
gamma_values = [100, 1000, 5000]


# Find the best parameters using 5-fold cross-validation
best_accuracy = 0

for c in C_values:
    for gamma in gamma_values:
        accuracy = train_and_evaluate_svm(data, c, gamma)
        if accuracy > best_accuracy:
            best_c = c
            best_gamma = gamma
            best_accuracy = accuracy
            #best_params.append((best_c, best_gamma, best_accuracy))

            
print("-----------------------------------------------------")
# Train the SVM model with the best parameters
print('Training SVM with best parameters: C = {}, gamma = {}, accuracy = {}%'.format(best_c, best_gamma, best_accuracy))
#train_and_evaluate_svm(data, best_c, best_gamma)
print("-----------------------------------------------------")
# Evaluate the SVM with and without scaling
print('Evaluating SVM with best parameters and scaling')
accuracy_with_scaling = train_and_evaluate_svm(data, best_c, best_gamma)
print('With scaling, accuracy = {:.2f}%'.format(accuracy_with_scaling))
print("-----------------------------------------------------")
print('Evaluating SVM with best parameters without scaling')
accuracy_without_scaling = train_and_evaluate_svm(data, best_c, best_gamma, scaling=False)
print('Without scaling, accuracy = {:.2f}%'.format(accuracy_without_scaling))

