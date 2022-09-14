# Common configurations for other python/jupyter files
# For repeated iteration, please refer to the SHELL (.sh) or POWERSHELL files (.ps1)
import platform

# Training-Size
# The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. 
# There are 50000 training images and 10000 test images. 
num_train = 25000
num_test  = 5000
num_classes = 10

# Set parameters for CNN-Training
batch_size = 32
num_epochs = 5
num_of_units = 256

# True/False: Run hyper-parameter search via Keras-Tuner or GridSearchCV
hyper_parameter_search = True
# Keras-Tuner-only configurations
hps_max_epochs = 20      # Tests within testing sets
hps_max_trials = 20      # Complete testing sets

# For echo operating system parameters
os = platform.platform()
cpu = platform.processor()
