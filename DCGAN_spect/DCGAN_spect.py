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
    
def normal_init(m, mean, std):
    
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
        
def show_result(num_epoch, show = False, save = False, path = 'result.png'):

    G.eval()
    
    noise = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
    
    test_images = G(noise)

    G.train()

    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5*5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ax[i, j].imshow(test_images[k, 0].cpu().data.numpy(), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()
        
def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()
        
# training parameters
batch_size = 128
lr = 0.0002
train_epoch = 8

# data_loader
img_size = 64
transform = transforms.Compose([
        transforms.Scale(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True)

# network
G = generator(16)
D = discriminator(16)
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)
G.cpu()
D.cpu()

# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()

# Adam optimizer
Gen_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
Disc_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# results save folder
if not os.path.isdir('MNIST_DCGANSP_results'):
    os.mkdir('MNIST_DCGANSP_results')
if not os.path.isdir('MNIST_DCGANSP_results/Samples'):
    os.mkdir('MNIST_DCGANSP_results/Samples')

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []
num_iter = 0

print('training start!')
start_time = time.time()
for epoch in range(train_epoch):
    
    Disc_losses = []
    Gen_losses = []
    epoch_start_time = time.time()
    
    for real, _ in train_loader:
          
        # train discriminator D
        D.zero_grad()

        mini_batch = real.size()[0]

        y_real_ = torch.ones(mini_batch)
        y_fake_ = torch.zeros(mini_batch)

        real, y_real_, y_fake_ = Variable(real.cpu()), Variable(y_real_.cpu()), Variable(y_fake_.cpu())
        
        Disc_result = D(real).squeeze()
        
        Disc_real_loss = BCE_loss(Disc_result, y_real_)

        noise = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
        noise = Variable(noise.cpu())
        Gen_result = G(noise)

        Disc_result = D(Gen_result).squeeze()
        Disc_fake_loss = BCE_loss(Disc_result, y_fake_)
        Disc_fake_score = Disc_result.data.mean()

        Disc_train_loss = Disc_real_loss + Disc_fake_loss

        Disc_train_loss.backward()
        Disc_optimizer.step()

        # Graph
        Disc_losses.append(Disc_train_loss.item())

        # train generator G
        G.zero_grad()

        noise_2 = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
        noise_2 = Variable(noise_2.cpu())

        Gen_result = G(noise_2)
        Disc_result = D(Gen_result).squeeze()
        Gen_train_loss = BCE_loss(Disc_result, y_real_)
        Gen_train_loss.backward()
        Gen_optimizer.step()
        
        # Graph
        Gen_losses.append(Gen_train_loss.item())

        num_iter += 1

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time


    print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(Disc_losses)),
                                                              torch.mean(torch.FloatTensor(Gen_losses))))
    
    # show samples and save 
    p = 'MNIST_DCGANSP_results/Samples/MNIST_DCGAN_' + str(epoch + 1) + '.png'

    show_result((epoch+1), show = True, save=True, path=p)
    
    # Graphing
    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(Disc_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(Gen_losses)))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print("Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
print("Training finish!... save training results")

#params save
torch.save(G.state_dict(), "MNIST_DCGANSP_results/generator_param.pkl")
torch.save(D.state_dict(), "MNIST_DCGANSP_results/discriminator_param.pkl")


