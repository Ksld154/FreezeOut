# VGG net stolen from the TorchVision package.
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import math
import numpy as np

from utils import scale_fn


class Layer(nn.Module):
    def __init__(self, n_in, n_out, layer_index, padding=0):
        super(Layer, self).__init__()

        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size=5,
                               padding=padding, bias=False)
        self.layer_index = layer_index

        # If the layer is being trained or not
        self.active = True

    def forward(self, x):
        if not self.active:
            self.eval()
        out = F.relu(self.conv1(x))
        if self.active:
            return out
        else:
            return out.detach()


class LinearLayer(nn.Module):
    def __init__(self, n_in, n_out, layer_index):
        super(LinearLayer, self).__init__()

        self.linear = nn.Linear(in_features=n_in, out_features=n_out)
        self.layer_index = layer_index

        # If the layer is being trained or not
        self.active = True

    def forward(self, x):
        if not self.active:
            self.eval()
        out = F.relu(self.linear(x))
        # print(out.shape)

        if self.active:
            return out
        else:
            return out.detach()


class Model(nn.Module):

    def __init__(self, growthRate, depth, nClasses, epochs, t_0, scale_lr=True, how_scale='cubic', const_time=False):
        super(Model, self).__init__()

        self.epochs = epochs
        self.t_0 = t_0
        self.scale_lr = scale_lr
        self.how_scale = how_scale
        self.const_time = const_time

        self.features = self.make_layers()
        self.fc = self.make_fc()
        self.layer_index = 5

        self._initialize_weights()

        # for m in self.modules():
        #     print(m)

        # Optimizer
        self.optim = optim.SGD([{
            'params': m.parameters(), 'lr': m.lr, 'layer_index': m.layer_index}
            for m in self.modules() if hasattr(m, 'active')]
        )
        # Iteration Counter
        self.j = 0

        # A simple dummy variable that indicates we are using an iteration-wise
        # annealing scheme as opposed to epoch-wise.
        self.lr_sched = {'itr': 0}

    def forward(self, x):
        x = self.features(x)

        x = x.view(x.size(0), -1)
        x = F.log_softmax(self.fc(x), dim=-1)
        return x

    def make_layers(self):
        layers = []

        layers += [Layer(n_in=3, n_out=6, layer_index=0, padding=0)]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        layers += [Layer(n_in=6, n_out=16, layer_index=1, padding=0)]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        return nn.Sequential(*layers)

    def make_fc(self):
        fc_layers = []
        fc_layers += [LinearLayer(n_in=16 * 5 * 5, n_out=120, layer_index=2)]
        fc_layers += [LinearLayer(n_in=120, n_out=84, layer_index=3)]
        fc_layers += [LinearLayer(n_in=84, n_out=10, layer_index=4)]
        return nn.Sequential(*fc_layers)

    def update_lr(self):

        # Loop over all modules
        for m in self.modules():

            # If a module is active:
            if hasattr(m, 'active') and m.active:

                # If we've passed this layer's freezing point, deactivate it.
                # print(self.j)
                if self.j > m.max_j:
                    m.active = False

                    # Also make sure we remove all this layer from the optimizer
                    for i, group in enumerate(self.optim.param_groups):
                        if group['layer_index'] == m.layer_index:
                            self.optim.param_groups.remove(group)

                # If not, update the LR
                else:
                    for i, group in enumerate(self.optim.param_groups):
                        if group['layer_index'] == m.layer_index:
                            self.optim.param_groups[i]['lr'] = (
                                m.lr/2)*(1+np.cos(np.pi*self.j/m.max_j))
        # Update the iteration counter
        self.j += 1

    def _initialize_weights(self):
        self.layer_cnt = 0
        for m in self.modules():
            # if isinstance(m, nn.Conv2d):
            #     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            #     # m.weight.data.normal_(0, math.sqrt(2. / n))
            #     if m.bias is not None:
            #         m.bias.data.zero_()
            # elif isinstance(m, nn.BatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     n = m.weight.size(1)
            #     # m.weight.data.normal_(0, 0.01)
            #     m.bias.data.zero_()

            # Set the layerwise scaling and annealing parameters
            if hasattr(m, 'active'):
                self.layer_cnt += 1
                m.lr_ratio = scale_fn[self.how_scale](
                    self.t_0 + (1 - self.t_0) * float(m.layer_index) / (self.layer_index-1))
                m.max_j = self.epochs * math.ceil(45000.0/64.0) * m.lr_ratio
                # print(m.layer_index)
                # print(m.max_j)

                if self.layer_cnt == self.layer_index:
                    pass

                # Optionally scale the learning rates to have the same total
                # distance traveled (modulo the gradients).
                m.lr = 0.1 / m.lr_ratio if self.scale_lr else 0.1
                print(
                    f'Layer #{m.layer_index}, learning rate: {m.lr:.4f}, stop iterations: {int(m.max_j)}, lr_ratio: {m.lr_ratio:.4f}')
                # print(m.lr_ratio)
