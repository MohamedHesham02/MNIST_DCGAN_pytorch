import os
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

class generator(nn.Module):
    # initializers
    def __init__(self, input_size=32, n_class = 10):
        super(generator, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc1_bn = nn.BatchNorm1d(128*4)
        self.fc2 = nn.Linear(self.fc1.out_features, 128)
        self.fc3 = nn.Linear(self.fc2.out_features, n_class)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.fc1_bn(self.fc1(input)), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.tanh(self.fc3(x))

        return x
        
class discriminator(nn.Module):
    # initializers
    def __init__(self, input_size=32, n_class=10):
        super(discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(self.fc1.out_features, 512)
        self.fc3 = nn.Linear(self.fc2.out_features, n_class)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.fc1(input), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.sigmoid(self.fc3(x))

        return x


# load weights

D = discriminator(input_size=28*28, n_class=1).eval()
G = generator(input_size=100, n_class=28*28).eval()

D.load_state_dict(torch.load('MNIST_GANBN_results/discriminator_param.pkl'))
G.load_state_dict(torch.load('MNIST_GANBN_results/generator_param.pkl'))

batch_size = 15

noise = torch.randn((batch_size, 100))

fake_images = G(noise)

fake_images_np = fake_images.cpu().detach().numpy()
fake_images_np = fake_images_np.reshape(fake_images_np.shape[0], 28, 28)

Row, Col = 3, 5

for i in range(batch_size):
    plt.subplot(Row, Col, i + 1)
    plt.imshow(fake_images_np[i], cmap='gray')
    
plt.show()
