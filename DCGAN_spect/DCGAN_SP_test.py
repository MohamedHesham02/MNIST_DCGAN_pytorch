import os, time
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
from torch.nn import Parameter
from torch.optim.optimizer import Optimizer, required
from torch import Tensor


def l2_normalize(v):
    return v / (v.norm() + 1e-12)

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        
        if not self.params_1st_Method():
            self.params_2nd_Method()
            
    def params_1st_Method(self):
        
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_w")
            
            return True
        
        except AttributeError:
            return False


    def params_2nd_Method(self):
        
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        
        u.data = l2_normalize(u.data)
        v.data = l2_normalize(v.data)
        w = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_w", w)

    def update_u_v(self):
        
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_w")
        height = w.data.shape[0]
        
        for i in range(self.power_iterations):
            
            v.data = l2_normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2_normalize(torch.mv(w.view(height,-1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))


    def forward(self, *args):
        self.update_u_v()
        return self.module.forward(*args)
        

class generator(nn.Module):
    # initializers
    def __init__(self, d):
        super(generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(100, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 1, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.tanh(self.deconv5(x))

        return x
        
class discriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(discriminator, self).__init__()
        self.conv1 = SpectralNorm(nn.Conv2d(1, d, 4, 2, 1))
        self.conv2 = SpectralNorm(nn.Conv2d(d, d*2, 4, 2, 1))
        self.conv3 = SpectralNorm(nn.Conv2d(d*2, d*4, 4, 2, 1))
        self.conv4 = SpectralNorm(nn.Conv2d(d*4, d*8, 4, 2, 1))
        self.conv5 = SpectralNorm(nn.Conv2d(d*8, 1, 4, 1, 0))

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = F.sigmoid(self.conv5(x))

        return x



# load weights

D = discriminator(16).eval()
G = generator(16).eval()

D.load_state_dict(torch.load('MNIST_DCGANSP_results/discriminator_param.pkl'))
G.load_state_dict(torch.load('MNIST_DCGANSP_results/generator_param.pkl'))

batch_size = 15

noise = torch.randn(batch_size, 100, 1, 1)

fake_images = G(noise)

fake_images_np = fake_images.cpu().detach().numpy()
fake_images_np = fake_images_np.reshape(fake_images_np.shape[0], 64, 64)

Row, Col = 3, 5

for i in range(batch_size):
    plt.subplot(Row, Col, i + 1)
    plt.imshow(fake_images_np[i], cmap='gray')
    
plt.show()
