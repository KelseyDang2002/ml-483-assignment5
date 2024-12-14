import numpy as np
import matplotlib.pyplot as plt
import time
import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as AF
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

### CNN architecture
class CNN(nn.Module):
    # The probability of dropout, number of hidden nodes, number of output classes
    def __init__(self, dropout_pr, num_hidden, num_classes):
        super(CNN, self).__init__()
        # Conv2d is for two dimensional convolution (which means input image is 2D)
        # Conv2d(in_channels=, out_channels=, kernel_size=, stride=1, padding=0)
        # in_channels=1 if grayscale, 3 for color; out_channels is # of output channels (or # of kernels)
        # Ouput size for W from CONV = (W-F+2P)/S+1 where W=input_size, F=filter_size, S=stride, P=padding
        # max_pool2d(input_tensor, kernel_size, stride=None, padding=0) for 2D max pooling
        # Output size for W from POOL = (W-F)/S+1 where S=stride=dimension of pool
        # K is # of channels for convolution layer; D is # of channels for pooling layer
        # Layer 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0) # K=D=10, output_size W = ((28-5)/1)+1=24 (24x24), (default S=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # W = ((24-2)/2)+1=12 (12x12), S=2 (pool dimension) since no overlapping regions
        self.dropout_conv1 = nn.Dropout2d(dropout_pr) # to avoid overfitting by dropping some nodes
        
        #+ You can add more convolutional and pooling layers
        # Layer 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0) # W = ((12-5)/1)+1=8 (8x8)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # W = ((8-2)/2)+1= (4x4)
        self.dropout_conv2 = nn.Dropout2d(dropout_pr)
        
        # Fully connected layer after convolutional and pooling layers
        self.num_flatten_nodes = 32*4*4 # Flatten nodes from 32 channels and 4*4 pool_size = 32*4*4 = 512
        self.fc1 = nn.Linear(self.num_flatten_nodes, num_hidden)
        #+ You can add more hidden layers here if necessary
        self.out = nn.Linear(num_hidden, num_classes) # the output nodes are 10 classes (10 digits)
        
    def forward(self, x):
        out = AF.relu(self.pool1(self.conv1(x)))
        out = AF.relu(self.dropout_conv1(out))

        out = AF.relu(self.pool2(self.conv2(out)))
        out = AF.relu(self.dropout_conv2(out))

        out = out.view(-1, self.num_flatten_nodes) # flattening
        out = AF.relu(self.fc1(out))
        out = AF.dropout(out) # Apply dropout for the randomly selected nodes by zeroing out before output during training
        output = self.out(out)
        return output

# To display some images
def show_some_digit_images(images):
    print("> Shapes of image:", images.shape)
    #print("Matrix for one image:")
    #print(images[1][0])
    for i in range(0, 10):
        plt.subplot(2, 5, i+1) # Display each image at i+1 location in 2 rows and 5 columns (total 2*5=10 images)
        plt.imshow(images[i][0], cmap='Oranges') # show ith image from image matrices by color map='Oranges'
    plt.show()

# Training function
def train_CNN_model(num_epochs, training_data, device, CUDA_enabled, CNN_model, loss_func, optimizer):
    train_losses = []
    CNN_model.train() # to set the model in training mode. Only Dropout and BatchNorm care about this flag.
    for epoch_cnt in range(num_epochs):
        for batch_cnt, (images, labels) in enumerate(training_data):
            # Each batch contain batch_size (100) images, each of which 1 channel 28x28
            # print(images.shape) # the shape of images=[100,1,28,28]
            # So, we need to flatten the images into 28*28=784

            if (device.type == 'cuda' and CUDA_enabled):
                images = images.to(device) # moving tensors to device
                labels = labels.to(device)

            optimizer.zero_grad() # set the cumulated gradient to zero
            output = CNN_model(images) # feedforward images as input to the network
            loss = loss_func(output, labels) # computing loss
            #print("Loss: ", loss)
            #print("Loss item: ", loss.item())
            train_losses.append(loss.item())
            # PyTorch's Autograd engine (automatic differential (chain rule) package) 
            loss.backward() # calculating gradients backward using Autograd
            optimizer.step() # updating all parameters after every iteration through backpropagation

            # Display the training status
            if (batch_cnt+1) % mini_batch_size == 0:
                print(f"Epoch={epoch_cnt+1}/{num_epochs}, batch={batch_cnt+1}/{num_train_batches}, loss={loss.item()}")
    return train_losses

# Testing function
def test_CNN_model(device, CUDA_enabled, CNN_model, testing_data):
    # torch.no_grad() is a decorator for the step method
    # making "require_grad" false since no need to keeping track of gradients    
    predicted_digits=[]
    # torch.no_grad() deactivates Autogra engine (for weight updates). This help run faster
    with torch.no_grad():
        CNN_model.eval() # # set the model in testing mode. Only Dropout and BatchNorm care about this flag.
        for batch_cnt, (images, labels) in enumerate(testing_data):
            if (device.type == 'cuda' and CUDA_enabled):
                images = images.to(device) # moving tensors to device
                labels = labels.to(device)
            
            output = CNN_model(images)
            _, prediction = torch.max(output,1) # returns the max value of all elements in the input tensor
            predicted_digits.append(prediction)
            num_samples = labels.shape[0]
            num_correct = (prediction==labels).sum().item()
            accuracy = num_correct/num_samples
            if (batch_cnt+1) % mini_batch_size == 0:
                print(f"batch={batch_cnt+1}/{num_test_batches}")
        print(f"> Number of samples= {num_samples}, Number of correct predictions = {num_correct}, Accuracy = {accuracy}")
    return predicted_digits

########################### Checking GPU and setup #########################
### CUDA is a parallel computing platform and toolkit developed by NVIDIA. 
# CUDA enables parallelize the computing intensive operations using GPUs.
# In order to use CUDA, your computer needs to have a CUDA supported GPU and install the CUDA Toolkit
# Steps to verify and setup Pytorch and CUDA Toolkit to utilize your GPU in your machine:
# (1) Check if your computer has a compatible GPU at https://developer.nvidia.com/cuda-gpus
# (2) If you have a GPU, continue to the next step, else you can only use CPU and ignore the rest steps.
# (3) Downloaded the compatible Pytorch version and CUDA version, refer to https://pytorch.org/get-started/locally/
# Note: If Pytorch and CUDA versions are not compatible, Pytorch will not be able to recognize your GPU
# (4) The following codes will verify if Pytorch is able to recognize the CUDA Toolkit:
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if (torch.cuda.is_available()):
    print("The CUDA version is", torch.version.cuda)
    # Device configuration: use GPU if available, or use CPU
    cuda_id = torch.cuda.current_device()
    print("ID of the CUDA device:", cuda_id)
    print("The name of the CUDA device:", torch.cuda.get_device_name(cuda_id))
    print("GPU will be utilized for computation.")
else:
    print("CUDA is supported in your machine. Only CPU will be used for computation.")
#exit()

############################### modeling #################################
### Convert the image into numbers: transforms.ToTensor()
# It separate the image into three color channels RGB and converts the pixels of each images to the brightness
# of the color in the range [0,255] that are scaled down to a range [0,1]. The image is now a Torch Tensor (array object)
### Normalize the tensor: transforms.Normalize() normalizes the tensor with mean (0.5) and stdev (0.5)
#+ You can change the mean and stdev values
print("------------------ANN modeling---------------------------")
transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,)),])
# PyTorch tensors are like NumPy arrays that can run on GPU
# e.g., x = torch.randn(64,100).type(dtype) # need to cast tensor to a CUDA datatype (dtype)

from torch.autograd import Variable
x = Variable

### Download and load the dataset from the torch vision library to the directory specified by root=''
# MNIST is a collection of 7000 handwritten digits (in images) split into 60000 training images and 1000 for testing 
# PyTorch library provides a clean data set. The following command will download training data in directory './data'
train_dataset=datasets.MNIST(root='./data', train=True, transform=transforms, download=True)
test_dataset=datasets.MNIST(root='./data', train=False, transform=transforms, download=False)
print("> Shape of training data:", train_dataset.data.shape)
print("> Shape of testing data:", test_dataset.data.shape)
print("> Classes:", train_dataset.classes)

# You can use random_split function to splite a dataset
#from torch.utils.data.dataset import random_split
#train_data, val_data, test_data = random_split(train_dataset, [60,20,20])

### DataLoader will shuffle the training dataset and load the training and test dataset
mini_batch_size = 50 #+ You can change this mini_batch_size
# If mini_batch_size==100, # of training batches=6000/100=600 batches, each batch contains 100 samples (images, labels)
# DataLoader will load the data set, shuffle it, and partition it into a set of samples specified by mini_batch_size.
train_dataloader=DataLoader(dataset=train_dataset, batch_size=mini_batch_size, shuffle=True)
test_dataloader=DataLoader(dataset=test_dataset, batch_size=mini_batch_size, shuffle=False)
num_train_batches = len(train_dataloader)
num_test_batches = len(test_dataloader)
print("> Mini batch size: ", mini_batch_size)
print("> Number of batches loaded for training: ", num_train_batches)
print("> Number of batches loaded for testing: ", num_test_batches)

### Let's display some images from the first batch to see what actual digit images look like
iterable_batches = iter(train_dataloader) # making a dataset iterable
images, labels = next(iterable_batches) # If you can call next() again, you get the next batch until no more batch left
show_digit_image = True
if show_digit_image:
    show_some_digit_images(images)

### Create an object for the ANN model defined in the class
# Architectural parameters: You can change these parameters except for num_input and num_classes
num_input = 28*28   # 28X28=784 pixels of image *** DON'T CHANGE ***
num_classes = 10    # output layer *** DON'T CHANGE ***
num_hidden = 10     # number of neurons at the first hidden layer
# Randomly selected neurons by dropout_pr probability will be dropped (zeroed out) for regularization.
dropout_pr = 0.05

#exit()

# CNN model
CNN_model = CNN(dropout_pr, num_hidden, num_classes)
print("> CNN model parameters")
print(CNN_model.parameters)

# To turn on/off CUDA if I don't want to use it.
CUDA_enabled = True
if (device.type == 'cuda' and CUDA_enabled):
    print("...Modeling using GPU...")
    CNN_model = CNN_model.to(device=device)
else:
    print("...Modeling using CPU...")

### Define a loss function: You can choose other loss functions
loss_func = nn.CrossEntropyLoss()

### Choose a gradient method
# model hyperparameters and gradient methods
# optim.SGD performs gradient descent and update the weigths through backpropagation.
num_epochs = 1
alpha = 0.06       # learning rate
gamma = 0.7        # momentum

# Stochastic Gradient Descent (SGD) is used in this program.
#+ You can choose other gradient methods (Adagrad, adadelta, Adam, etc.) and parameters
# CNN optimizer
CNN_optimizer = optim.SGD(CNN_model.parameters(), lr=alpha, momentum=gamma) 

### Train your networks
print("............Training CNN................")
start = time.time()
train_loss=train_CNN_model(num_epochs, train_dataloader, device, CUDA_enabled, CNN_model, loss_func, CNN_optimizer)
end = time.time()
print(f"\n> Training Time: {end - start} seconds\n")
print(f"> Loss Function: Cross Entropy Loss")
print(f"> Gradient Method: SGD")
print(f"> Learning Rate: {alpha}")
print(f"> Momentum (Gamma): {gamma}\n")

print("............Testing CNN model................")
predicted_digits=test_CNN_model(device, CUDA_enabled, CNN_model, test_dataloader)
print("> Predicted digits by CNN model")
print(predicted_digits)

#### To save and load models and model's parameters ####
