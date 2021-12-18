# Alphabet Soup Donation Program

## Overview

Alphabet Soup, a philanthropic foundation dedicated to helping organizations that help the environment, improve people's overall wellbeing, and unify the world, has raised and donated over $10 Billion to oganizations in the last 20 years.  Organizations have used these donations to invest in life-saving technologies and organize reforestation groups around the world.  Alphabet Soup realizes the impact their donations can have and seeks to give to those companies that are most likely to put these donations to good use.  Alphabet Soup's CEO has asked their data scientist group to analyze the company's past donation pool (a pool of 34,000 organizations) and develop a mathematical, data driven tool to pinpoint the best companies to provide donations.  The data scientist team has determined they will utilize an advanced statistical modeling technique such as the deep learning Neural Network to guide Alphabet Soup in their future donations.

For this project, the data scientist team conducted the following:
* Preprocessed Data for a Neural Network Model
* Compiled, Trained, and Evaluated the Model
* Saved model weights and saved the results to file
* Optimized the Model and made comparisons to other machine learning models

## Results

### Data Preprocessing
*  ***Target Variable*** - For the target, the datascientist team has identified a datapoint which evaluated the company's success in implementing a project implemented using Alphabet Soup's donation.  In the dataset this variable is labeled "IS_SUCCESSFUL".
*  ***Features Variables*** - Variables that were considered features are essentially every column except the target, IS_SUCCESSFUL.  Note the columns that were not considered as features below.
*  ***Inconsequential Variables*** - Variables which were determined as inconsequential and not having an affect on determining the best donation candidates were: EIN - employer identification number and NAME - the company name.

### Compiling, Training, and Evaluating the Model
***Neurons, layers, and activation function:*** The neural network model used 2 hidden layers: the first layer had 80 neurons and the second layer had 30 neurons.  An output layer is also utilized. The first and second hidden layers utilized the Rectified Linear Unit (ReLU) activation function and the activation function for the output layer is the sigmoid function.

***Target Model Performance*** The model performance ("first NN trial") had a loss: 70% and accuracy: 70%.  The model did not achieve the Target Model Performance rate of 75%.  Further models are tested below to potentially approach the Target Model Performance of 75%.  The actual evaluation statistics the first NN trial model are: 

* Performance:
268/268 - 0s - loss: 0.6958 - accuracy: 0.6968 - 229ms/epoch - 855us/step; Loss: 0.6958094835281372, Accuracy: 0.6967930197715759

### Saved the Model Weights
The model weights are saved every 5 epochs and the results are saved to an HDF5 file.

Epoch 1/100
804/804 [==============================] - 1s 943us/step - loss: 0.5349 - accuracy: 0.7414
Epoch 2/100
804/804 [==============================] - 1s 968us/step - loss: 0.5351 - accuracy: 0.7413
Epoch 3/100
804/804 [==============================] - 1s 955us/step - loss: 0.5345 - accuracy: 0.7414
Epoch 4/100
804/804 [==============================] - 1s 978us/step - loss: 0.5343 - accuracy: 0.74130s - loss: 0.5342 - accuracy
Epoch 5/100
776/804 [===========================>..] - ETA: 0s - loss: 0.5346 - accuracy: 0.7403
Epoch 00005: saving model to checkpoints\weights.05.hdf5
804/804 [==============================] - 1s 1ms/step - loss: 0.5341 - accuracy: 0.7408
Epoch 6/100
804/804 [==============================] - 1s 942us/step - loss: 0.5337 - accuracy: 0.7409
Epoch 7/100
804/804 [==============================] - 1s 922us/step - loss: 0.5340 - accuracy: 0.7413
Epoch 8/100
804/804 [==============================] - 1s 905us/step - loss: 0.5344 - accuracy: 0.7413
Epoch 9/100
804/804 [==============================] - 1s 954us/step - loss: 0.5341 - accuracy: 0.74140s - loss: 0.5352 - accuracy: 0.
Epoch 10/100
739/804 [==========================>...] - ETA: 0s - loss: 0.5354 - accuracy: 0.7404 ETA: 0s - loss: 0.5335 - accuracy
Epoch 00010: saving model to checkpoints\weights.10.hdf5

### Optimizing the Model
Since the model fell short of the target performance of 75%, several varying techniques were evaluated to potentially approach the target.  These techniques included the following:

***1.  Increased Neurons from 80 to 150 for the first layer and from 30 to 75 for the second layer.  This trial proved to yield very similar yet unimproved results (accuracy rate of 68%).  To simplify processing time and reduce iterations, the decision was made to keep the number of neurons the same as the ***first NN trial***.***
   
Model: "sequential_2"
_________________________________________________________________
 Layer (type)                Output Shape              Param    
=================================================================
 dense_8 (Dense)             (None, 150)               6750      
                                                                 
 dense_9 (Dense)             (None, 75)                11325     
                                                                 
 dense_10 (Dense)            (None, 1)                 76        
                                                                 
=================================================================
Total params: 18,151
Trainable params: 18,151
Non-trainable params: 0
_________________________________________________________________

Performance:
268/268 - 0s - loss: 1.1529 - accuracy: 0.6838 - 353ms/epoch - 1ms/step
Loss: 1.152901530265808, Accuracy: 0.6838483810424805

***2.  Increased Hidden Layers from 2 to 4.  The first layer has 80 neurons, the second has 30 neurons, the third has 20 neurons, and the fourth has 20 neurons. This trial proved to yield very similar yet unimproved results (accuracy rate of 68%).  To simplify processing time and reduce iterations, the decision was made to keep the number of hidden layers the same as the ***first NN trial***.***

Model: "sequential_3"
_________________________________________________________________
 Layer (type)                Output Shape              Param    
=================================================================
 dense_11 (Dense)            (None, 80)                3600      
                                                                 
 dense_12 (Dense)            (None, 30)                2430      
                                                                 
 dense_13 (Dense)            (None, 20)                620       
                                                                 
 dense_14 (Dense)            (None, 20)                420       
                                                                 
 dense_15 (Dense)            (None, 1)                 21        
                                                                 
=================================================================
Total params: 7,091
Trainable params: 7,091
Non-trainable params: 0
_________________________________________________________________

Performance
268/268 - 0s - loss: 1.1529 - accuracy: 0.6838 - 353ms/epoch - 1ms/step
Loss: 1.152901530265808, Accuracy: 0.6838483810424805

***3.  Changed the activation function of each hidden layer from ReLU to Leaky ReLU.  Kept the output layer activation function the same with sigmoid.  Again, the change yielded very similar results with an accuracy rate of 68% and therefore, stay with the ***first NN trial***.***

First hidden layer:
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim = number_input_features,activation = "leaky_relu"))
Second hidden layer:
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2,activation="leaky_relu"))
Third hidden layer:
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer3,activation="leaky_relu"))
Third hidden layer:
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer4,activation="leaky_relu"))
Output layer:
nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

Performance:
268/268 - 0s - loss: 1.1529 - accuracy: 0.6838 - 353ms/epoch - 1ms/step
Loss: 1.152901530265808, Accuracy: 0.6838483810424805

***4.  Changed the model altogether to the Logistic Regression model.  The Logistic regression model accuracy is 47%.  This is a very low performance and cannot be utilized over any other model.***

log_classifier = LogisticRegression(solver = "lbfgs", max_iter = 200)
log_classifier.fit(X_train,y_train)
y_pred = log_classifier.predict(X_test)
print(f"Logistic regression model accuracy: {accuracy_score(y_test,y_pred):0.3f}")

***5.  Again, changed the model altogether to the machine learning Random Forest Classifier.  The Random Forest Classifier predictive accuracy is a solid 71% and matches the accuracy of the ***first NN trial***.***

rf_model = RandomForestClassifier(n_estimators=128, random_state=78)
rf_model = rf_model.fit(X_train_scaled,y_train)
y_pred = rf_model.predict(X_test_scaled)
print(f" Random Forest Classifier predictive accuracy: {accuracy_score(y_test,y_pred):0.3f}")

***6.  Removed the "USE_CASE" columns in an attempt to reduce noise; however, this attempt at 68% did not increase the accuracy rate to target.  The results from this option are as follows:***

Performance:
268/268 - 0s - loss: 1.1529 - accuracy: 0.6838 - 353ms/epoch - 1ms/step
Loss: 1.152901530265808, Accuracy: 0.6838483810424805

## Summary
It is unfortunate that none of the Neuron Network trial modifications nor selection of the alternative statistical models reached the target model performance of 75%.  With that, the data scientist team determined the best model to potentially use is The Random Forest Classifier. 

The ***first NN trial*** results yielded 70% accuracy.   While the Random Forest Classifier's results at 71% were not materially improved the model is simpler and easier to administer than the Neural Network model.  Albeit, there are times when a Random Forest Classifier should not be used over a Neuron Network model (as with image data or natural language data, for example); however, in this case the data is tabular, and can be trained with a sufficient number of estimators and tree depth so that the Random Forest performance can match that of the deep learning neuron network.  Also in this case the Random Forest Classifier runs in seconds versus the Neural Network model's minutes run time; and therefore, Random Forest Classifier is the best model at this point in time.
