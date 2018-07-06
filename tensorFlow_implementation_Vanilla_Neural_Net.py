import numpy as np
import tensorflow as tf

# 1. Generation of fake input data - Our X and Y would need to be be either numpy array or a dict of numpy arrays.

# X1 is a "n x D" [i.e. 100 x 2] multi-dimensional array; X1 will belong to class 1
X_class1_feature1 = np.random.normal(100,30,(100))  
X_class1_feature2 = np.random.normal(800,60,(100))

# Similarly, we have X2 and X3
X_class2_feature1 = np.random.normal(500,50,(100))  
X_class2_feature2 = np.random.normal(300,40,(100))

X_class3_feature1 = np.random.normal(50,15,(100))  
X_class3_feature2 = np.random.normal(80,25,(100))

# Concatenate the features
X_feature1 = np.concatenate((X_class1_feature1,X_class2_feature1,X_class3_feature1)) # a Vector of size 300 [numpy array]
X_feature2 = np.concatenate((X_class1_feature2,X_class2_feature2,X_class3_feature2)) # a Vector of size 300 [numpy array]

# Build a dict of numpy arrays for X
X = {'feature1': X_feature1, 'feature2': X_feature2 }

# We will use a single numpy array for Y;  Y is vector of size 300 containing the labels [a numpy array by itself]
Y1 = np.zeros((100,1))
Y2 = np.ones((100,1))
Y3 = np.ones((100,1)) + 1
Y = np.concatenate((Y1,Y2,Y3))
Y = np.ndarray.flatten(Y)
Y = Y.astype(int)   # Cast each elements into int type     

# Creating an Input Function to feed Training Examples to the Estimator [An Estimator, introduced later, would need an Input Function which will feed the training examples into the Estimator]:
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x=X,
    y=Y,
    batch_size=100,
    num_epochs=None,
    shuffle=True)



# 2. Initialisation of variables

h = 3 # Our hidden layer will have 3 hidden units
K = 3 # K is no. of Classes

W1 = 0.01*np.random.rand(2,h)   # weights from Input Layer to Hidden Layer
b1 = np.zeros((1,h))   # bias units for the hidden layer

W2 = 0.01*np.random.randn(h,K) # weights from Hidden Layer to the Output Layer
b2 = np.zeros((1,K)) # bias units for the output layer

step_size = 0.001
reg = 0.001

# 3. Building a Model Function - A Model Function is a Function which defines the Model/Structure of our Neural Net; The Model Function is required by the Estimator[Estimator will be introduced later].


def vanilla_neural_model_fn(features, labels, mode):    # A Model Function has a fixed set of parameters and return value that it must accept and return so that it could be used by an Estimator.

    # Input Layer - To be a Tensor with a shape of [ batch_size, num_of_dif_features]
    f1_numeric_feature_column = tf.feature_column.numeric_column(key="feature1")
    f2_numeric_feature_column = tf.feature_column.numeric_column(key="feature2")
    columns = [f1_numeric_feature_column, f2_numeric_feature_column]
    input_layer = tf.feature_column.input_layer( features, columns)  # Have a Shape of [ batch_size, num_of_dif_features] i.e. [100, 2]

    # Hidden Layer [Implemented by a TensorFlow's "Dense Layer"]
    hidden_layer = tf.layers.dense(inputs = input_layer, units = 3, activation=tf.nn.relu)

    # Output Layer [also uses a "Dense Layer"]
    output_layer = tf.layers.dense(inputs=hidden_layer, units = 3)   # This layer essentially outputs the score elements for each training example
    predictions = {
                "classes": tf.argmax(output_layer,axis=1),         # get the predicted classes
                "probabilities": tf.nn.softmax(output_layer,name="softmax_tensor")   # applying softmax func on the score elements to get the predicted probabilities
                   }

    # If in Prediction mode:
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)   # if the purpose is to just predict, then the model function ends here. 


    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=output_layer) # Using the Cross Entropy Loss
    

    # If in Training mode:
    if mode == tf.estimator.ModeKeys.TRAIN:

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=step_size)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
    
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    

    # If in Evaluation mode 
    eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
    
    return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)




# 4. Doing the Training using an Estimator

# Create the Estimator ['Estimator' is a TensorFlow class used for performing 1.training of model, 2.evaluation of model, and 3.inference using the model]
neural_classifier = tf.estimator.Estimator(model_fn=vanilla_neural_model_fn, model_dir="/tmp/vanilla_model")

# Training the Model - To Train the Model, we need to use the Estimator. And the Estimator would need to be given an Input Function that would feed the Training Examples to the Estimator.
neural_classifier.train(input_fn=train_input_fn, steps=1000 )
        




# 5. Display of results
print("Training completed")
print()
# Generate a Test Example from Class 1
test_X_class1_feature1 = np.random.normal(100,30,(1))  
test_X_class1_feature2 = np.random.normal(800,60,(1))

# Generate a Test Example from Class 2
test_X_class2_feature1 = np.random.normal(500,50,(1))  
test_X_class2_feature2 = np.random.normal(300,40,(1))

# Generate a Test Example from Class 3
test_X_class3_feature1 = np.random.normal(50,15,(1))  
test_X_class3_feature2 = np.random.normal(80,25,(1))

test_X_feature1 = np.concatenate((test_X_class1_feature1,test_X_class2_feature1,test_X_class3_feature1)) # a Vector of size 300 [numpy array]
test_X_feature2 = np.concatenate((test_X_class1_feature2,test_X_class2_feature2,test_X_class3_feature2)) # a Vector of size 300 [numpy array]

# Build a dict of numpy arrays for X_test
test_X = {'feature1': test_X_feature1, 'feature2': test_X_feature2 }

test_Y1 = np.zeros((1,1))
test_Y2 = np.ones((1,1)) 
test_Y3 = np.ones((1,1)) + 1
test_Y = np.concatenate((test_Y1,test_Y2,test_Y3))
test_Y = np.ndarray.flatten(test_Y)
test_Y = test_Y.astype(int)       

test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x=test_X,
    y=test_Y,
    num_epochs=1,
    shuffle=False)
test_results = list(neural_classifier.predict(input_fn=test_input_fn))




print("Testing a Random Example from Class 1...")
print("Input is ")
print(str(test_X_class1_feature1)+" "+str(test_X_class1_feature2))
print()
print("Output is Class ")
print(str(test_results[0]["classes"] +1))     # plus 1 since our Class starts from Class 1 instead of Class 0
print()

print("Testing a Random Example from Class 2...")
print("Input is ")
print(str(test_X_class2_feature1)+" "+str(test_X_class2_feature2))
print()
print("Output is Class ")
print(str(test_results[1]["classes"] +1))

print()
print("Testing a Random Example from Class 3...")
print("Input is ")
print(str(test_X_class3_feature1)+" "+str(test_X_class3_feature2))
print()
print("Output is Class ")
print(str(test_results[2]["classes"] +1))    
    
    
    
    





