import torch
import torch.nn as nn
import math
import torch.nn.functional as F


def get_embedder(multires, i=1):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': i,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

class LipschitzLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, zero=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.zeros((out_features, in_features), device="cuda", requires_grad=True)) + 1e-6
        self.bias = torch.nn.Parameter(torch.zeros((out_features), device="cuda", requires_grad=True))
        self.c = torch.nn.Parameter(torch.zeros((1), device="cuda", requires_grad=True))
        self.softplus = torch.nn.Softplus()

        self.zero = zero
        self.initialize_parameters()

    def initialize_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        if not self.zero:
            self.weight.data.uniform_(-stdv, stdv)
            self.bias.data.uniform_(-stdv, stdv)

        W = self.weight.data
        W_abs_row_sum = torch.abs(W).sum(1)
        self.c.data = W_abs_row_sum.max() 

    def get_lipschitz_constant(self):
        return self.softplus(self.c)

    def forward(self, input):
        lipc = self.softplus(self.c)
        scale = lipc / torch.abs(self.weight).sum(1)
        scale = torch.clamp(scale, max=1.0)
        return torch.nn.functional.linear(input, self.weight * scale.unsqueeze(1), self.bias)


embed_C_fn, C_input_ch = get_embedder(6, 1)
    
class MLPLip(nn.Module):
    def __init__(self, input_size, output_size, target=""):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.target = target
        self.slope = 0.01

        self.lrelu = nn.LeakyReLU(self.slope)

        W = 48  
        self.main = nn.Sequential(
                nn.Linear(self.input_size + C_input_ch, W*2),
                nn.Linear(W*2, W*2),
                nn.Linear(W*2, W*2),
            )

        self.x_head = nn.Sequential(LipschitzLinear(W*2, W*2),
                                    LipschitzLinear(2*W, W),
                                    LipschitzLinear(W, 3, zero=False))
        
        self.c_head = nn.Sequential(LipschitzLinear(W*2, W*2),
                                    LipschitzLinear(2*W, W),
                                    LipschitzLinear(W, 48, zero=False))

        self.o_head = nn.Sequential(LipschitzLinear(W*2, W*2),
                                    LipschitzLinear(2*W, W),
                                    LipschitzLinear(W, 1, zero=True))

        self.r_head = nn.Sequential(LipschitzLinear(W*2, W*2),
                                    LipschitzLinear(2*W, W),
                                    LipschitzLinear(W, 4, zero=False))

        self.s_head = nn.Sequential(LipschitzLinear(W*2, W*2),
                                    LipschitzLinear(2*W, W),
                                    LipschitzLinear(W, 3, zero=False))


    def forward(self, x, rotations, scales, means, opacity, c=1.0, target="xc"): 
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.normalize(x)
        means = torch.nn.functional.normalize(means)

        C = torch.ones(opacity.shape, requires_grad=False).fill_(c).cuda()
        C_emb = embed_C_fn(C)

        x = torch.concat([x, rotations, scales, means, opacity, C_emb], dim=1)

        if c == 0.:
            x = x.detach()

        for ii in range(len(self.main)):
            x = self.lrelu(self.main[ii](x))

        deta_x = 0.
        if 'x' in target:
            x_x = x
            for ii in range(len(self.x_head)):
                if ii != len(self.x_head) - 1 :
                    x_x = self.lrelu(self.x_head[ii](x_x)) 
                else:
                    x_x = self.x_head[ii](x_x)
            deta_x = x_x

        deta_c = 0.
        if 'c' in target:
            x_c = x
            for ii in range(len(self.c_head)):
                if ii != len(self.c_head) - 1 :
                    x_c = self.lrelu(self.c_head[ii](x_c)) 
                else:
                    x_c = self.c_head[ii](x_c)
            deta_c = x_c.view(-1, 16, 3)

        deta_o = 0.
        if 'o' in target:
            x_o = x
            for ii in range(len(self.o_head)):
                if ii != len(self.o_head) - 1 :
                    x_o = self.lrelu(self.o_head[ii](x_o)) 
                else:
                    x_o = self.o_head[ii](x_c)
            deta_o = x_o  

        deta_r = 0.
        if 'r' in target:
            x_r = x
            for ii in range(len(self.r_head)):
                if ii != len(self.r_head) - 1 :
                    x_r = self.lrelu(self.r_head[ii](x_r)) 
                else:
                    x_r = self.r_head[ii](x_r)
            deta_r = x_r  

        deta_s = 0.
        if 's' in target:
            x_s = x
            for ii in range(len(self.s_head)):
                if ii != len(self.s_head) - 1 :
                    x_s = self.lrelu(self.s_head[ii](x_s)) 
                else:
                    x_s = self.s_head[ii](x_s)
            deta_s = x_s  

        return deta_x, deta_r,  deta_s, deta_c, deta_o

    def get_lipschitz_loss(self, target="xc"):
        loss_lip = 1.
        if 'x' in target:
            for ii in range(len(self.x_head)):
                loss_lip = loss_lip * self.x_head[ii].get_lipschitz_constant()
        if 'c' in target:
            for ii in range(len(self.c_head)):
                loss_lip = loss_lip * self.c_head[ii].get_lipschitz_constant()
        if 'o' in target:
            for ii in range(len(self.o_head)):
                loss_lip = loss_lip * self.o_head[ii].get_lipschitz_constant()  
        if 'r' in target:
            for ii in range(len(self.r_head)):
                loss_lip = loss_lip * self.r_head[ii].get_lipschitz_constant()    
        if 's' in target:
            for ii in range(len(self.s_head)):
                loss_lip = loss_lip * self.s_head[ii].get_lipschitz_constant()       
        return loss_lip

