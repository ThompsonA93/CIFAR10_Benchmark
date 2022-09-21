### Packages
from datetime import datetime
import time
import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 22
plt.rc('font', size=MEDIUM_SIZE)            # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)        # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)        # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)       # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)       # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)       # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)     # fontsize of the figure title

# Dataset
from keras.datasets import cifar10

# Subroutines
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
import keras_tuner as kt
from sklearn.metrics import confusion_matrix

# Additional configurations, @see config.py
import config

### Configurations
# Training-Size
num_train = config.num_train                   # 60000 for full data set 
num_test  = config.num_test                    # 10000 for full data set

# Simple functions to log information
path = os.getcwd()+"/log"
logDir = os.path.exists(path)
if not logDir:
    os.makedirs(path)

plots = os.getcwd()+"/log/plots"
logDir = os.path.exists(plots)
if not logDir:
    os.makedirs(plots)

training_results = path+"/keras-nn-training-log.txt"
def log_training_results(*s):
    with open(training_results, 'a') as f:
        for arg in s:
            print(arg, file=f)
            print(arg)

hyperparameter_search_log = path+"/keras-nn-hyperparameter-tuning-log.txt"
def log_hyperparameter_search(*s):
    with open(hyperparameter_search_log, 'a') as f:
        for arg in s:
            print(arg, file=f)
            print(arg)

print("Generated data will be located in ", training_results, hyperparameter_search_log)
print("Generated plots will be located in ", plots)

log_training_results("[%s] on (%s, %s) using (Train: %s, Test: %s)" % (datetime.now(), config.os, config.cpu, config.num_train, config.num_test))
if config.hyper_parameter_search:
    log_hyperparameter_search("[%s] on (%s, %s) using (Train: %s, Test: %s)" % (datetime.now(), config.os, config.cpu, config.num_train, config.num_test))


# Fetch CIFAR10-Data from Keras repository
(X_train, y_train), (X_test, y_test) = cifar10.load_data()


print("\t\t\t\t (Sets,  X,  Y, RGB)")
print("Shape of training data:\t\t", X_train.shape)
print("Shape of training labels:\t", y_train.shape)
print("Shape of testing data:\t\t", X_test.shape)
print("Shape of testing labels:\t", y_test.shape)

# Visualize some examples
cols=8
rows=4
fig, ax = plt.subplots(ncols=cols, nrows=rows, figsize=(cols*2, rows*2))
index = 0
for i in range(rows):
    for j in range(cols):
        ax[i,j].imshow(X_train[index])
        ax[i,j].set_title(y_train[y_train[index][0]], fontsize=16)
        ax[i,j].imshow(X_train[index])
        ax[i,j].axis('off')
        index += 1
plt.show(block = False)
fig.savefig(plots+'/cifar10_examples.png')

train_data = X_train
train_label = y_train
test_data = X_test
test_label = y_test

# Reshape the data such that we have access to every pixel of the image
train_data = X_train.astype('float32')
train_label = y_train.astype("float32")
test_data = X_test.astype('float32')
test_label = y_test.astype("float32")

# We know the RGB color code where different values produce various colors. It is also difficult to remember every color combination. 
# We already know that each pixel has its unique color code and also we know that it has a maximum value of 255. 
# To perform Machine Learning, it is important to convert all the values from 0 to 255 for every pixel to a range of values from 0 to 1.
train_data = train_data / 255
test_data = test_data / 255

# Categorize the labels by conversion from integers to a class matrix
train_label = keras.utils.to_categorical(train_label, config.num_classes)
test_label = keras.utils.to_categorical(test_label, config.num_classes)

# As an optional step, we decrease the training and testing data size, such that the algorithms perform their execution in acceptable time
train_data = train_data[1:num_train,]
train_label = train_label[1:num_train]

test_data = test_data[1:num_test,]
test_label = test_label[1:num_test]

print("\t\t\t\t (Sets,  X, Y, RGB )")
print("Reshaped training data:\t\t", train_data.shape)
print("Reshaped training labels:\t", train_label.shape)
print("Reshaped testing data:\t\t", test_data.shape)
print("Reshaped testing labels:\t", test_label.shape)

### Create model: https://keras.io/guides/sequential_model/
model = Sequential()

### Minimum Model
#model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(32,32,3), activation='relu', padding='same')) 
#model.add(Flatten()) 
#model.add(Dense(units=config.num_classes, activation="softmax"))

### Standard model
# First convolutional layer
model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(32,32,3), activation='relu', padding='same')) 
# Second convolutional layer
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
# Max pooling layer
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Dropout(0.25)) 
# Flatten input into feature vector and feed into dense layer
model.add(Flatten()) 
model.add(Dense(units=config.num_of_units, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=config.num_of_units, activation='relu'))
model.add(Dropout(0.5))
# Outputs from dense layer are projected onto 10 unit output layer
model.add(Dense(units=config.num_classes, activation="softmax"))

# Compile model
optimizer = keras.optimizers.RMSprop(
    learning_rate=0.0001,
    epsilon = 1e-6,
)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()

# Train model

start_time = time.time()
history = model.fit(
    x=train_data, 
    y=train_label, 
    batch_size=config.batch_size, 
    epochs=config.num_epochs, 
    shuffle=True, 
    validation_data=(test_data, test_label))
end_time = time.time() - start_time

params = {"Keras":{"batch_size":config.batch_size, "epochs":config.num_epochs}}
log_training_results("--- [%s] Trained new model: %s in %s seconds ---" % (datetime.now(), params, end_time))

fig = plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.show(block = False)
fig.savefig(plots+'/training_history_standard.png')


# Evaluate model based on supplied tags
start_time = time.time()
test_loss, test_acc = model.evaluate(train_data, train_label)
end_time = time.time() - start_time

log_training_results("\tPredicting train data -- Execution time: %ss; Accuracy: %s; Loss: %s" % (end_time, test_acc, test_loss))  

# Evaluate model based on supplied tags
start_time = time.time()
test_loss, test_acc = model.evaluate(test_data, test_label)
end_time = time.time() - start_time

log_training_results("\tPredicting test data -- Execution time: %ss; Accuracy: %s; Loss: %s" % (end_time, test_acc, test_loss))

# Let model predict data
y_pred = model.predict(test_data)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(test_label, axis=1)

# Visualize estimation over correct and incorrect prediction via confusion matrix
confusion_mtx = confusion_matrix(y_true, y_pred_classes)
fig, ax = plt.subplots(figsize=(16,8))
ax = sns.heatmap(confusion_mtx, annot=True, fmt='d', ax=ax, cmap="Blues")
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_title('CIFAR-10 Keras Confusion Matrix of standard NN')
plt.show(block = False)
fig.savefig(plots+'/ConfusionMatrix_standard.png')

if not config.hyper_parameter_search:
    print("Terminating without hyperparameter search.")
    exit(0)
print("Starting hyperparameter search over %s epochs each" % (config.hps_max_epochs))

""""
# Overengineered Model
# Takes too damn long
# Debatable whether this might produce any meaningful data
# Gives a fair overview over what can be configured for HPS
def model_builder(hp):
    ### Base Model
    model = keras.Sequential()

    ### Layers    
    # MANDATORY :: First convolutional layer
    hp_first_conv2d_activation = hp.Choice('First Conv2D activation', ['relu', 'tanh', 'sigmoid'])
    model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(32,32,3), activation=hp_first_conv2d_activation, padding='same')) 

    # OPTIONAL :: Second convolutional layer
    hp_second_conv2d = hp.Boolean("Use Second Conv2D")
    if hp_second_conv2d:
        hp_second_conv2d_activation = hp.Choice('Second Conv2D activation', ['relu', 'tanh', 'sigmoid'])
        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation=hp_second_conv2d_activation))

    # OPTIONAL :: Max pooling layer
    hp_maxpooling2d = hp.Boolean("Use MaxPooling2D")
    if hp_maxpooling2d:
        model.add(MaxPooling2D(pool_size=(2, 2))) 
        
    # OPTIONAL :: Dropout
    hp_first_dropout = hp.Boolean("Use First Dropout")
    if hp_first_dropout:
        hp_first_dropout_rate = hp.Float('First dropout rate', min_value=0.25, max_value=0.50, step=0.05)
        model.add(Dropout(rate=hp_first_dropout_rate))

    # MANDATORY :: Flatten input into feature vector and feed into dense layer
    model.add(Flatten()) 
    
    # OPTIONAL :: First dense layer 
    hp_first_dense = hp.Boolean("Use First Dense")
    if hp_first_dense:
        hp_first_dense_activation = hp.Choice('First dense activation', ['relu', 'tanh', 'sigmoid'])
        hp_first_dense_units = hp.Int('First dense units', min_value=32, max_value=512, step=32)
        model.add(Dense(units=hp_first_dense_units, activation=hp_first_dense_activation))
    
    # OPTIONAL :: Dropout
    hp_second_dropout = hp.Boolean("Use Second Dropout")
    if hp_second_dropout:
        hp_second_dropout_rate = hp.Float('Second dropout rate', min_value=0.25, max_value=0.50, step=0.05)
        model.add(Dropout(rate=hp_second_dropout_rate))
        
    # OPTIONAL :: Second dense layer
    hp_second_dense = hp.Boolean("Use Second Dense")
    if hp_second_dense:
        hp_second_dense_activation = hp.Choice('Second dense activation', ['relu', 'tanh', 'sigmoid'])
        hp_second_dense_units = hp.Int('Second dense units', min_value=32, max_value=512, step=32)
        model.add(Dense(units=hp_second_dense_units, activation=hp_second_dense_activation))

    # OPTIONAL :: Dropout
    hp_third_dropout = hp.Boolean("Use Third Dropout")
    if hp_third_dropout:
        hp_third_dropout_rate = hp.Float('Third dropout rate', min_value=0.25, max_value=0.50, step=0.05)
        model.add(Dropout(rate=hp_third_dropout_rate))

    # MANDATORY :: Outputs from dense layer are projected onto 10 unit output layer
    hp_third_dense_activation = hp.Choice('Third activation', ['relu', 'tanh', 'sigmoid'])
    model.add(Dense(units=config.num_classes, activation=hp_third_dense_activation))

    # Compile model
    hp_learning_rate = hp.Choice('Optimizer learnrate', [1e-2, 1e-3, 1e-4, 1e-5])
    optimizer = keras.optimizers.RMSprop(
        learning_rate=hp_learning_rate,
        epsilon = 1e-6,
    )
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model
"""

# Hypermodel
def model_builder(hp):
    # Base Model
    model = keras.Sequential()

    # Hyperparameters
    hp_secondary_conv2d = hp.Boolean("Use 2nd Conv. Layer")
    hp_conv_activations = hp.Choice('Conv2D Activation', ['relu', 'tanh', 'sigmoid'])

    hp_use_maxpooling2d = hp.Boolean("Use MaxPooling2D")

    hp_dropout = hp.Boolean("Use dropout layers")
    hp_dropout_rate = hp.Float('Dropout rate', min_value=0.25, max_value=0.50, step=0.05)

    hp_use_dense = hp.Boolean("Use dense layers")
    hp_dense_units = hp.Int('Dense Units', min_value=32, max_value=512, step=32)
    hp_dense_activations = hp.Choice('Dense Activation', ['relu', 'tanh', 'sigmoid'])

    hp_learning_rate = hp.Choice('Learning rate', [1e-2, 1e-3, 1e-4, 1e-5])
    
    # First convolutional layer
    model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(32,32,3), activation=hp_conv_activations, padding='same')) # Usually RELU

    # Second convolutional layer
    if hp_secondary_conv2d:
        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation=hp_conv_activations)) # Usually RELU

    # Max pooling layer
    if hp_use_maxpooling2d:
        model.add(MaxPooling2D(pool_size=(2, 2))) 

    if hp_dropout:
        model.add(Dropout(rate=hp_dropout_rate))

    # Flatten input into feature vector and feed into dense layer
    model.add(Flatten()) 
    
    if hp_use_dense:
        model.add(Dense(units=hp_dense_units, activation=hp_dense_activations))

    if hp_dropout:
        model.add(Dropout(rate=hp_dropout_rate))
        
    #model.add(Dense(units=config.num_of_units, activation='relu'))
    if hp_use_dense:
        model.add(Dense(units=hp_dense_units, activation=hp_dense_activations))

    if hp_dropout:
        model.add(Dropout(rate=hp_dropout_rate))
    
    # Outputs from dense layer are projected onto 10 unit output layer
    model.add(Dense(units=config.num_classes, activation=hp_dense_activations)) # Usually softmax

    # Compile model
    optimizer = keras.optimizers.RMSprop(
        learning_rate=hp_learning_rate,
        epsilon = 1e-6,
    )

    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

tuner = kt.RandomSearch(
    model_builder,
    objective='val_accuracy',
    #max_epochs=config.max_trials,
    #factor=3,                    
    directory='log/hps',
    project_name='keras-hyperparameter-search-RandomSearch'
)

stop_early = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, verbose=1)
csvlogger = keras.callbacks.CSVLogger(hyperparameter_search_log, separator=",", append=True)
tuner.search(
    train_data,
    train_label,
    epochs=config.hps_max_epochs,
    validation_split=0.2,
    callbacks=[stop_early]
)

log_hyperparameter_search("--- [%s] Running Parameter-Tests [Keras-RandomSearch] ---" % datetime.now())
best_hps_rs = tuner.get_best_hyperparameters(num_trials=1)[0]
log_hyperparameter_search("Best parameters set found on following development set: %s" % best_hps_rs.values)
best_hps_rs_results = tuner.results_summary(num_trials=1)
best_hps_rs_model = tuner.get_best_models(num_models=1)[0]
rs_test_loss, rs_test_acc = best_hps_rs_model.evaluate(test_data, test_label)
log_hyperparameter_search("\tPredicting test data -- Accuracy: %s; Loss: %s" % (rs_test_acc, rs_test_loss))  

top3_hps = tuner.get_best_hyperparameters(3)
top3_models = tuner.get_best_models(3)
for i in range(1, 3):
    log_hyperparameter_search("Additional parameters set found on following development set: %s" % top3_hps[i].values)
    loss, acc = top3_models[i].evaluate(test_data, test_label)
    log_hyperparameter_search("\tPredicting test data -- Accuracy: %s; Loss: %s" % (acc, loss))  

tuner = kt.BayesianOptimization(
    model_builder,
    objective='val_accuracy',
    max_trials=config.hps_max_trials,
    #factor=3,                    
    directory='log/hps',
    project_name='keras-hyperparameter-search-BayesianOptimization'
)

stop_early = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, verbose=1)
csvlogger = keras.callbacks.CSVLogger(hyperparameter_search_log, separator=",", append=True)
tuner.search(
    train_data,
    train_label,
    epochs=config.hps_max_epochs,
    validation_split=0.2,
    callbacks=[stop_early]
)

log_hyperparameter_search("--- [%s] Running Parameter-Tests [Keras-BayesianOptimization] ---" % datetime.now())
best_hps_bo = tuner.get_best_hyperparameters(num_trials=1)[0]
log_hyperparameter_search("Best parameters set found on following development set: %s" % best_hps_bo.values)

best_hps_bo_results = tuner.results_summary(num_trials=1)

best_hps_bo_model = tuner.get_best_models(num_models=1)[0]
bo_test_loss, bo_test_acc = best_hps_bo_model.evaluate(test_data, test_label)
log_hyperparameter_search("\tPredicting test data -- Accuracy: %s; Loss: %s" % (bo_test_acc, bo_test_loss))  

top3_hps = tuner.get_best_hyperparameters(3)
top3_models = tuner.get_best_models(3)
for i in range(1, 3):
    log_hyperparameter_search("Additional parameters set found on following development set: %s" % top3_hps[i].values)
    loss, acc = top3_models[i].evaluate(test_data, test_label)
    log_hyperparameter_search("\tPredicting test data -- Accuracy: %s; Loss: %s" % (acc, loss))  

tuner = kt.Hyperband(
    model_builder,
    objective='val_accuracy',
    max_epochs=config.hps_max_trials,
    factor=3,                    
    directory='log/hps',
    project_name='keras-hyperparameter-search-Hyperband'
)

stop_early = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, verbose=1)
csvlogger = keras.callbacks.CSVLogger(hyperparameter_search_log, separator=",", append=True)
tuner.search(
    train_data,
    train_label,
    epochs=config.hps_max_epochs,
    validation_split=0.2,
    callbacks=[stop_early]
)

log_hyperparameter_search("--- [%s] Running Parameter-Tests [Keras-Hyperband] ---" % datetime.now())
best_hps_hb = tuner.get_best_hyperparameters(num_trials=1)[0]
log_hyperparameter_search("Best parameters set found on following development set: %s" % best_hps_hb.values)

best_hps_hb_results = tuner.results_summary(num_trials=1)

best_hps_hb_model = tuner.get_best_models(num_models=1)[0]
hb_test_loss, hb_test_acc = best_hps_hb_model.evaluate(test_data, test_label)
log_hyperparameter_search("\tPredicting test data -- Accuracy: %s; Loss: %s" % (hb_test_acc, hb_test_loss))  

top3_hps = tuner.get_best_hyperparameters(3)
top3_models = tuner.get_best_models(3)
for i in range(1, 3):
    log_hyperparameter_search("Additional parameters set found on following development set: %s" % top3_hps[i].values)
    loss, acc = top3_models[i].evaluate(test_data, test_label)
    log_hyperparameter_search("\tPredicting test data -- Accuracy: %s; Loss: %s" % (acc, loss))  

# Map and print the scores
accuracy_score = {"RandomSearch": rs_test_acc, "BayesianOptimization": bo_test_acc, "HyperBand": hb_test_acc}
loss_score = {"RandomSearch": rs_test_loss, "BayesianOptimization": bo_test_loss, "HyperBand": hb_test_loss}

print("--- Finalized scores ")
print("\tAccuracy: %s" % accuracy_score)
print("\tLoss: %s" % loss_score)

# Choose the best model out of all three based on accuracy
# For Loss, it is required to adjust the score here and the values taken for the diagrams below.
# Tip: Search ''accuracy'' and replace
best_accuracy = max(accuracy_score, key=accuracy_score.get)
print("Algorithm: %s" % best_accuracy)

best_accuracy_value = max(accuracy_score.values())
print("Score: %s" % best_accuracy_value)

model = 0
if best_accuracy == "RandomSearch":
    model = tuner.hypermodel.build(best_hps_rs)
elif best_accuracy == "BayesianOptimization":
    model = tuner.hypermodel.build(best_hps_bo)
elif best_accuracy == "HyperBand":
    model = tuner.hypermodel.build(best_hps_hb)


# Train a new model with the optimal algorithm and parameters
start_time = time.time()
history = model.fit(
    x=train_data, 
    y=train_label, 
    batch_size=config.batch_size, 
    epochs=config.num_epochs, 
    shuffle=True, 
    validation_data=(test_data, test_label)
)
end_time = time.time() - start_time
params = {"HPS-Opt-Keras":{"batch_size":config.batch_size, "epochs":config.num_epochs}}
log_training_results("--- [%s] Trained new model: %s in %s seconds ---" % (datetime.now(), params, end_time))

# Display Accuracy
fig = plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.show(block = False)
fig.savefig(plots+'/training_history_optimal.png')

# When a machine learning model has high training accuracy and very low validation then this case is probably known as over-fitting. The reasons for this can be as follows:
#    The hypothesis function you are using is too complex that your model perfectly fits the training data but fails to do on test/validation data.
#    The number of learning parameters in your model is way too big that instead of generalizing the examples , your model learns those examples and hence the model performs badly on test/validation data.

start_time = time.time()
test_loss, test_acc = model.evaluate(train_data, train_label)
end_time = time.time() - start_time
log_training_results("\tPredicting train data -- Execution time: %ss; Accuracy: %s; Loss: %s" % (end_time, test_acc, test_loss))  

# Evaluate model based on supplied tags
start_time = time.time()
test_loss, test_acc = model.evaluate(test_data, test_label)
end_time = time.time() - start_time

log_training_results("\tPredicting test data -- Execution time: %ss; Accuracy: %s; Loss: %s" % (end_time, test_acc, test_loss))  

# Let model predict data
y_pred = model.predict(test_data)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(test_label, axis=1)

# Visualize estimation over correct and incorrect prediction via confusion matrix
confusion_mtx = confusion_matrix(y_true, y_pred_classes)
fig, ax = plt.subplots(figsize=(16,8))
ax = sns.heatmap(confusion_mtx, annot=True, fmt='d', ax=ax, cmap="Blues")
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_title('CIFAR-10 Keras Confusion Matrix of optimal NN')
plt.show(block = False)
fig.savefig(plots+'/ConfusionMatrix_optimal.png')