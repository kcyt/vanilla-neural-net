import numpy as np

# 1. Generation of fake input data


# X1 is a "n x D" [i.e. 100 x 2] multi-dimensional array; X1 will belong to class 1
X_class1_feature1 = np.random.normal(100,30,(100))  
X_class1_feature2 = np.random.normal(800,60,(100))
X1 = np.array([X_class1_feature1,X_class1_feature2])
X1 = X1.T

# Similarly, we have X2 and X3
X_class2_feature1 = np.random.normal(500,50,(100))  
X_class2_feature2 = np.random.normal(300,40,(100))
X2 = np.array([X_class2_feature1,X_class2_feature2])
X2 = X2.T

X_class3_feature1 = np.random.normal(50,15,(100))  
X_class3_feature2 = np.random.normal(80,25,(100))
X3 = np.array([X_class3_feature1,X_class3_feature2])
X3 = X3.T

X = np.concatenate((X1,X2,X3))  # X is 300 x 2 array
y1 = np.ones((100,1))
y2 = np.ones((100,1)) + 1
y3 = np.ones((100,1)) + 2
y = np.concatenate((y1,y2,y3))  # y is 300 x 1 vector of labels



# 2. Initialisation of variables

h = 3 # Our hidden layer will have 3 hidden units
K = 3 # K is no. of Classes

W1 = 0.01*np.random.rand(2,h)   # weights from Input Layer to Hidden Layer
b1 = np.zeros((1,h))   # bias units for the hidden layer

W2 = 0.01*np.random.randn(h,K) # weights from Hidden Layer to the Output Layer
b2 = np.zeros((1,K)) # bias units for the output layer

step_size = 0.001
reg = 0.001



# 3. Doing Gradient Descent in a loop

print("Working on Gradient Descent...")
print()
for i in range(1000):
    hidden_layer = np.maximum(0,np.dot(X,W1)+b1) # using ReLU Activation Function
    scores = np.dot(hidden_layer, W2) +b2
    exp_scores = np.exp(scores)
    probs = exp_scores/np.sum(exp_scores,axis=1,keepdims=True)
    loss_for_each_example = -np.log(probs[range(300), ((y-1).T).astype(int)])  # y-1 because our class starts from 1 instead of 0.
    data_loss = np.sum(loss_for_each_example)/300   # get the average loss across all the examples
    reg_loss = 0.5 * reg * np.sum(W1*W1) + 0.5* reg* np.sum(W2*W2)   # Note W1*W1 is element-wise multiplication
    total_loss = data_loss + reg_loss

    # Compute the gradients
    dscores = probs   # dscores refer to dL/dscores where L is the total_loss
    dscores[range(300),((y-1).T).astype(int) ] -= 1
    dscores /= 300

    dW2 = np.dot(hidden_layer.T,dscores) + reg*W2   # dW2 refers to dL/dW2
    db2 = np.sum(dscores, axis=0, keepdims=True)    # db2 refers to dL/db2; likewise for the rest below
    dhidden = np.dot(dscores,W2.T)
    dhidden[hidden_layer <= 0] = 0
    dW1 = np.dot(X.T, dhidden) + reg*W1
    db1 = np.sum(dhidden, axis=0, keepdims = True)

    W1 += -step_size * dW1
    b1 += -step_size * db1
    W2 += -step_size * dW2
    b2 += -step_size * db2

    # Progress reporting
    if i % 100 == 0:
        string = "Trained for " + str(i) + " steps"
        print(string)
        print("Loss is "+ str(total_loss))
        


# 4. Display of results
print("Training completed")
print()
# Generate a Test Example from Class 1
test_X_class1_feature1 = np.random.normal(100,30,(1))  
test_X_class1_feature2 = np.random.normal(800,60,(1))
test_X1 = np.array([test_X_class1_feature1,test_X_class1_feature2])
test_X1 = test_X1.T

# Generate a Test Example from Class 2
test_X_class2_feature1 = np.random.normal(500,50,(1))  
test_X_class2_feature2 = np.random.normal(300,40,(1))
test_X2 = np.array([test_X_class2_feature1,test_X_class2_feature2])
test_X2 = test_X2.T

# Generate a Test Example from Class 3
test_X_class3_feature1 = np.random.normal(50,15,(1))  
test_X_class3_feature2 = np.random.normal(80,25,(1))
test_X3 = np.array([test_X_class3_feature1,test_X_class3_feature2])
test_X3 = test_X3.T

test_X = np.concatenate((test_X1,test_X2,test_X3))  
hidden_layer = np.maximum(0, np.dot(test_X, W1) + b1)
scores = np.dot(hidden_layer, W2) + b2
predicted_class = np.argmax(scores, axis=1)

print("Testing a Random Example from Class 1...")
print("Input is ")
print(test_X1)
print()
print("Output is Class ")
print(str(predicted_class[0]+1))     # plus 1 since our Class starts from Class 1 instead of Class 0


print("Testing a Random Example from Class 2...")
print("Input is ")
print(test_X2)
print()
print("Output is Class ")
print(str(predicted_class[1]+1))


print("Testing a Random Example from Class 3...")
print("Input is ")
print(test_X3)
print()
print("Output is Class")
print(str(predicted_class[2]+1))    
    
    
    
    





