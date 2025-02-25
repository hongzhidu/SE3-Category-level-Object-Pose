import torch
import torch.nn as nn
from network.layers_equi import VNLinear
import torch.nn.functional as F

# Resnet Blocks
class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx




class DecoderInner(nn.Module):
    ''' Decoder class.

    It does not perform any form of normalization.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
    '''

    def __init__(self,  z_dim=128, hidden_size=512):
        super().__init__()
        self.z_dim = z_dim


        self.z_in = VNLinear(z_dim, z_dim)


        self.fc_in = nn.Linear(z_dim*3+1+6, hidden_size)

        self.block0 = ResnetBlockFC(hidden_size)
        self.block1 = ResnetBlockFC(hidden_size)
        self.block2 = ResnetBlockFC(hidden_size)
        self.block3 = ResnetBlockFC(hidden_size)
        self.block4 = ResnetBlockFC(hidden_size)

        self.fc_out = nn.Linear(hidden_size, 1)



    def forward(self, p, z, z_inv, cat_id):
        p = p.permute(0, 2, 1)

        batch_size, T, D = p.size()

        if cat_id.shape[0] == 1:
            obj_idh = cat_id.view(-1, 1).repeat(cat_id.shape[0], 1)
        else:
            obj_idh = cat_id.view(-1, 1)
        one_hot = torch.zeros(batch_size, 6).to(cat_id.device).scatter_(1, obj_idh.long(), 1)
        one_hot = one_hot.unsqueeze(2).repeat(1, 1, T).permute(0, 2, 1)  # (bs, N, cat_one_hot)

        net = torch.norm(p, dim=2, keepdim=True)

        z = z.view(batch_size, -1, D).contiguous()
        net_z = torch.einsum('bmi,bni->bmn', p, z)

        z_inv = z_inv.permute(0, 2, 1).repeat(1, T, 1)
        net = torch.cat([net, net_z, z_inv, one_hot], dim=2)


        net = self.fc_in(net)
        net = self.block0(net)
        net = self.block1(net)
        net = self.block2(net)
        net = self.block3(net)
        net = self.block4(net)

        out = self.fc_out(net)

        out = out.squeeze(-1)


        return out

