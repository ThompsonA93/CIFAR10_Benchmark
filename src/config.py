# Common configurations for other python/jupyter files
# For repeated iteration, please refer to the SHELL (.sh) or POWERSHELL files (.ps1)

import platform

# Training-Size
# The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. 
# There are 50000 training images and 10000 test images. 
num_train = 12500
num_test  = 2500
  
# Set parameters for CNN
batch_size = 32
num_epochs = 5
num_of_units = 256

# Use GridSearchCV to look up optimal parameters - Separate from actual training; takes a long time.
# True/False: Run hyper-parameter search via GridSearchCV. 
hyper_parameter_search = False      

# For echo operating system parameters
os = platform.platform()
cpu = platform.processor()
