from collections import OrderedDict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo

from models.sync_batchnorm import SynchronizedBatchNorm2d

def load_weights_sequential(target, source_state):
    
    new_dict = OrderedDict()
    for k1, v1 in target.state_dict().items():
        if not 'num_batches_tracked' in k1:
            tar_v = source_state[k1]

            if v1.shape != tar_v.shape:
                # Init the new segmentation channel with zeros
                # print(v1.shape, tar_v.shape)
                c, _, w, h = v1.shape
                tar_v = torch.cat([
                    tar_v, 
                    torch.zeros((c,3,w,h)),
                ], 1)

            new_dict[k1] = tar_v

    target.load_state_dict(new_dict)

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
}


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = SynchronizedBatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1, dilation=dilation)
        self.bn2 = SynchronizedBatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = SynchronizedBatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation,
                               padding=dilation, bias=False)
        self.bn2 = SynchronizedBatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = SynchronizedBatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers=(3, 4, 23, 3)):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = SynchronizedBatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                SynchronizedBatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x_1 = self.conv1(x)  # /2
        x = self.bn1(x_1)
        x = self.relu(x)
        x = self.maxpool(x)  # /2

        x_2 = self.layer1(x)
        x = self.layer2(x_2)   # /2
        x = self.layer3(x)
        x = self.layer4(x)

        return x, x_1, x_2


def resnet50(pretrained=True):
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        load_weights_sequential(model, model_zoo.load_url(model_urls['resnet50']))
    return model

def resnet101(pretrained=True):
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    if pretrained:
        load_weights_sequential(model, model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=True):
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    if pretrained:
        load_weights_sequential(model, model_zoo.load_url(model_urls['resnet152']))
    return model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Channel_Layernorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        return x

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
        
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
    
class Depthwise_Separable_Conv(nn.Module):
    def __init__(self, in_features,kernel_size,qkv_bias=False):
        super().__init__()
        self.dw = nn.Conv2d(in_features, in_features,kernel_size, 1, kernel_size//2, bias=qkv_bias, groups=in_features)
        self.bn = nn.BatchNorm2d(in_features)
        self.pw = nn.Conv2d(in_features, in_features,1, bias=qkv_bias)
        
    def forward(self, x):
        x = self.dw(x)
        x = self.bn(x)
        x = self.pw(x)
        return x
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features,1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features,1)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PS_Attention(nn.Module):
    def __init__(self, dim, kernel_size=3, pale_size=7,num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % 2 == 0, 'dim must be even'
        assert (dim//2) % num_heads == 0, 'dim should be divisible by num_heads'
        self.pale_size = pale_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.ModuleList([Depthwise_Separable_Conv(dim, kernel_size, qkv_bias),
                                  Depthwise_Separable_Conv(dim, kernel_size, qkv_bias),
                                  Depthwise_Separable_Conv(dim, kernel_size, qkv_bias)])
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
    
    def forward(self, x):
        assert x.shape[2]%self.pale_size == 0 & x.shape[3]%self.pale_size == 0, 'image size should be divisible by pale_size'
        
        q,k,v = self.qkv[0](x),self.qkv[1](x),self.qkv[2](x)
        
        q_h,q_w = q.tensor_split(2,dim=1)
        k_h,k_w = k.tensor_split(2,dim=1)
        v_h,v_w = v.tensor_split(2,dim=1)

        q_h,k_h,v_h = self.img2pale(q_h,x.shape[2],2),self.img2pale(k_h,x.shape[2],2),self.img2pale(v_h,x.shape[2],2)
        q_w,k_w,v_w = self.img2pale(q_w,x.shape[3],3),self.img2pale(k_w,x.shape[3],3),self.img2pale(v_w,x.shape[3],3)        

        x_h = self.pale2img(self.axis_attention(q_h,k_h,v_h,x.shape[0]),x.shape[2],2)
        x_w = self.pale2img(self.axis_attention(q_w,k_w,v_w,x.shape[0]),x.shape[3],3)
        
        x = torch.cat([x_h,x_w],dim=1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def axis_attention(self, q, k, v, B):
        B_,C,H,W = q.shape

        q = q.reshape([B_,C,-1]).reshape(B_, self.num_heads, C // self.num_heads, -1).permute(0,1,3,2)
        k = k.reshape([B_,C,-1]).reshape(B_, self.num_heads, C // self.num_heads, -1).permute(0,1,3,2)
        v = v.reshape([B_,C,-1]).reshape(B_, self.num_heads, C // self.num_heads, -1).permute(0,1,3,2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(2, 3).reshape([B_,C,-1]).reshape([B,-1,C,H,W])
        return x
    
    def img2pale(self, x, length, axis):
        x_idx = [torch.tensor(range(i,length,length//self.pale_size)) for i in range(length//self.pale_size)]
        x = torch.cat([x.index_select(axis,idx).unsqueeze(1) for idx in x_idx],dim = 1)
        x = x.reshape([-1]+list(x.shape[2:]))
        return x
    
    def pale2img(self, x, length, axis):
        x = sum([[x[:,j].index_select(axis,torch.tensor([i])) for j in range(length//self.pale_size)] for i in range(self.pale_size)],[])
        x = torch.cat(x,dim = axis)
        return x

class PS_Block(nn.Module):
    def __init__(self, dim, cpe_kernel_size=3, att_kernel_size=3, pale_size=7, num_heads=8, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0.,drop_path=0., act_layer=nn.GELU ,norm_layer=Channel_Layernorm):
        super().__init__()
        self.cpe = nn.Conv2d(dim, dim, cpe_kernel_size, 1, cpe_kernel_size//2, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = PS_Attention(dim, kernel_size=att_kernel_size, pale_size=pale_size, num_heads=num_heads, qkv_bias=qkv_bias, 
                                 qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
    def forward(self, x):
        x = self.cpe(x) + x
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
class PS_Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        in_dim = args['input_dim']
        self.patch_merg_convs = nn.ModuleList([])
        self.ps_blocks = nn.ModuleList([])
        for stage_arg in args['stage_args']:
            layers = nn.ModuleList([PS_Block(stage_arg['out_dim'],stage_arg['cpe_kernel_size'],
                                             stage_arg['att_kernel_size'],stage_arg['pale_size'],
                                             stage_arg['num_heads'],args['mlp_ratio'],args['qkv_bias'],
                                             args['qk_scale'],args['drop'],args['attn_drop'],
                                             args['drop_path'],args['act_layer'],
                                             args['norm_layer']) for i in range(stage_arg['num_block'])])
            self.ps_blocks.append(layers)
            self.patch_merg_convs.append(nn.Conv2d(in_dim, stage_arg['out_dim'], stage_arg['patch_merg_kernel_size'], 
                                                   stage_arg['patch_merg_stride_size'], stage_arg['patch_merg_kernel_size']//2))
            in_dim = stage_arg['out_dim']
        self.proj = nn.Linear(in_dim,args['num_classes'])
        self.softmax = nn.Softmax(1)
        
    def forward(self, x):
        for patch_merg_conv,blocks in zip(self.patch_merg_convs,self.ps_blocks):
            x = patch_merg_conv(x)
            for block in blocks:
                x = block(x)
        x = self.softmax(self.proj(x.mean([2, 3])))
        return x
      
stage_arg_1 = {
    'out_dim':128, 
    'patch_merg_kernel_size':7,
    'patch_merg_stride_size':4, 
    'cpe_kernel_size':3, 
    'att_kernel_size':3, 
    'pale_size':7,
    'num_heads':4,
    'num_block':2}

stage_arg_2 = {
    'out_dim':256, 
    'patch_merg_kernel_size':3,
    'patch_merg_stride_size':2,
    'cpe_kernel_size':3, 
    'att_kernel_size':3, 
    'pale_size':7,
    'num_heads':8, 
    'num_block':2}

stage_arg_3 = {
    'out_dim':512, 
    'patch_merg_kernel_size':3,
    'patch_merg_stride_size':2,
    'cpe_kernel_size':3, 
    'att_kernel_size':3, 
    'pale_size':7,
    'num_heads':16,
    'num_block':16}

stage_arg_4 = {
    'out_dim':1024, 
    'patch_merg_kernel_size':3,
    'patch_merg_stride_size':2, 
    'cpe_kernel_size':3, 
    'att_kernel_size':3, 
    'pale_size':7,
    'num_heads':32,
    'num_block':2}

args = {
    'input_dim':3,
    'num_classes':1000,
    'stage_args':[stage_arg_1,stage_arg_2,stage_arg_3,stage_arg_4],
    'mlp_ratio':4.,
    'qkv_bias':True,
    'qk_scale':None, 
    'drop':0., 
    'attn_drop':0.,
    'drop_path':0.,
    'act_layer':nn.GELU,
    'norm_layer':Channel_Layernorm}

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
  
def test():
  model = PS_Transformer(args)
  print(f'number of parameter:{get_n_params(model)}')
  model(torch.zeros(2,3,224,224))
  
if __name__ == "__main__":
  test()
"""