# EX 05: Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1. Initialization: Given two endpoints A and B, initialize the starting point to A (x₁, y₁).

2. Calculate Slope and Decision Variable: Compute the slope (m) of the line using the difference in y-coordinates (Δy) and x-coordinates (Δx). 

3. Initialize the decision variable (dᵢ) as 2Δy - Δx.
Pixel Selection:

4. If dᵢ < 0, choose the next pixel to the right (xᵢ₊₁ = xᵢ + 1).

5. If dᵢ ≥ 0, choose the next pixel to the right and up (xᵢ₊₁ = xᵢ + 1, yᵢ₊₁ = yᵢ + 1).

6. Update Decision Variable:

7. If the chosen pixel is to the right, update dᵢ as dᵢ = dᵢ + 2Δy.

8. If the chosen pixel is to the right and up, update dᵢ as dᵢ = dᵢ + 2Δy - 2Δx.

9. Repeat Steps 3 and 4:

10. Continue selecting pixels until reaching the endpoint B (x₂, y₂).

11. Plot the Line:
Plot the selected pixels to form the line from A to B.

## Program:
```python
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: VENKATANATHAN P R
RegisterNumber:  212223240173
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# reading the file
dataset=pd.read_csv('Placement_Data4.csv')
print("DATASET:\n")
print(dataset)

# dropping the serial no and salary col
dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)
# dataset=data.drop('salary',axis=1)

# categorising col for further labelling
dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
print("DATASET WITH TYPES:\n")
print(dataset.dtypes)

# labelling the columns
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes

# display dataset
print("DATASET:\n")
print(dataset)

#selecting the features and labels
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values

# display dependent variables
print("DEPENDENT VARIABLES:\n")
print(Y)

# Initialize the model parameters
theta=np.random.randn(X.shape[1])
y=Y

# Define the sigmoid function.
def sigmoid(z):
    return 1/(1+np.exp(-z))

# Define the loss function
def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))

# Define the gradient descent algorithm
def gradient_descent(theta,X,y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(X.dot(theta))
        gradient=X.T.dot(h-y)/m
        theta-=alpha*gradient
    return theta

# Train the model
theta=gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)

# Make predictions.
def predict(theta,X):
    h=sigmoid(X.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred

y_pred=predict(theta,X)

# Evaluate the model
accuracy=np.mean(y_pred.flatten()==y)
print("ACCURACY:\n",accuracy)

print("Y_PREDICTED:\n")
print(y_pred)

print("Y VALUES:\n")
print(Y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print("THE NEW Y_PREDICTED:\n")
print(y_prednew)

xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print("THE NEW Y_PREDICTED:\n")
print(y_prednew)

```

## Output:

### DATASET:

![alt text](<Screenshot 2024-05-08 235820.png>)

### DATASET WITH TYPES:

![alt text](<Screenshot 2024-05-08 235830.png>)

### DATASET:

![alt text](<Screenshot 2024-05-08 235841.png>)

### DEPENDENT VARIABLES:

![alt text](<Screenshot 2024-05-08 235853.png>)

### ACCURACY:

![alt text](<Screenshot 2024-05-08 235902.png>)

### Y_PREDICTED:

![alt text](<Screenshot 2024-05-08 235912.png>)

### Y VALUES:

![alt text](<Screenshot 2024-05-08 235921.png>)

### THE NEW Y_PREDICTED:

![alt text](<Screenshot 2024-05-08 235940.png>)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

