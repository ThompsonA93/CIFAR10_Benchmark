###############
# Init system #
############### 

# Load packages
library(keras)


# Disable warnings
options(warn=-1)


# Clear workspace
graphics.off()
rm(list=ls())


# Experimental setup
num_train = 5000
num_test  = 500


# Set parameters for CNN
batch_size = 32
num_epochs = 5
num_of_units = 256



#########################
# Load and prepare data #
#########################

# Load CIFAR-10 dataset
cifar10 = dataset_cifar10()


# Assign train and test data+labels
train_data = cifar10$train$x
test_data = cifar10$test$x

train_label = cifar10$train$y
test_label = cifar10$test$y


# Rescale data to range [0,1]
train_data = train_data / 255
test_data = test_data / 255


# Select subset for training
train_data = train_data[1:num_train,,,]
train_label = train_label[1:num_train]


# Select subset for testing
test_data = test_data[1:num_test,,,]
test_label = test_label[1:num_test]


# Create categorical labels
train_labelc = to_categorical(train_label, num_classes = 10)
test_labelc = to_categorical(test_label, num_classes = 10)



#####################################
# Define Keras network architecture #
#####################################

# Network architecture:
#    2 convolutional layers + max pooling layeer: feature extraction
#    2 fully connected layers: classification

# Init model
CNN <- keras_model_sequential()


# Define network architecture
CNN %>%
 
  # First convolutional layer
  layer_conv_2d( filter = 32, kernel_size = c(3,3), padding = "same", input_shape = c(32, 32, 3)) %>%
  layer_activation("relu") %>%

  # Second convolutional layer
  layer_conv_2d(filter = 32, kernel_size = c(3,3)) %>%
  layer_activation("relu") %>%

  # Max pooling layer
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%

  # Flatten input into feature vector and feed into dense layer
  layer_flatten() %>%
  layer_dense(num_of_units) %>%
  layer_activation("relu") %>%
  layer_dropout(0.5) %>%

  layer_dense(num_of_units) %>%
  layer_activation("relu") %>%
  layer_dropout(0.5) %>%

  # Outputs from dense layer are projected onto 10 unit output layer
  layer_dense(10) %>%
  layer_activation("softmax")


# Set parameters for optimizer
opt <- optimizer_rmsprop(lr = 0.0001, decay = 1e-6)


# Compile network
CNN %>% compile(
  loss = "categorical_crossentropy",
  optimizer = opt,
  metrics = "accuracy"
)


################################
# Train and run classification #
################################

# Init timer
t1 = proc.time()


# Train CNN
CNN %>% fit(
  train_data, train_labelc,
  batch_size = batch_size,
  epochs = num_epochs,
  shuffle = TRUE
)


cat("\n\nCorrect classification results (train data) :\n\n")

# Eval CNN on training data
result <- CNN %>% evaluate(train_data, train_labelc)
success =result[2]*100
res_s = sprintf('   Train: %5.2f\n\n', success)
cat(res_s)


cat("\n\nCorrect classification results (test data) :\n\n")

# Eval CNN on test data
result <- CNN %>% evaluate(test_data, test_labelc)
res_s = sprintf('    Test: %5.2f\n\n\n', success)
cat(res_s)


# End time, calculate elapsed time
t2 = proc.time()
t = t2-t1
cat("Computation time:\n\n")
print(t)
