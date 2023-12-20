# using mobileOne as the backbone to segment image!
# modified by Mr wang, Sun Yat-sen University, 2023-3-18!
# Email:1044625113@qq.com

import torch
import torch.nn as nn
try:
    from layers import MobileOneBlock
except:
    from AImodel.layers import MobileOneBlock
import torch.nn.functional as F



def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)
    
    
class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x
    

class FullyAttentionalBlock(nn.Module):
    """
    Fully Attentional Network for Semantic Segmentation,2022
    refer:https://github.com/Ilareina/FullyAttentional
    Modified by Mr Wangzhipan,2022-19-18
    """
    def __init__(self, plane, outplane):
        super(FullyAttentionalBlock, self).__init__()
        self.conv1 = nn.Linear(plane, plane)
        self.conv2 = nn.Linear(plane, plane)
        self.conv = nn.Sequential(nn.Conv2d(plane, outplane, 3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(outplane),
                                  nn.ReLU())

        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, _, height, width = x.size()

        feat_h = x.permute(0, 3, 1, 2).contiguous().view(batch_size * width, -1, height)
        feat_w = x.permute(0, 2, 1, 3).contiguous().view(batch_size * height, -1, width)
        encode_h = self.conv1(F.avg_pool2d(x, [1, width]).view(batch_size, -1, height).permute(0, 2, 1).contiguous())
        encode_w = self.conv2(F.avg_pool2d(x, [height, 1]).view(batch_size, -1, width).permute(0, 2, 1).contiguous())

        energy_h = torch.matmul(feat_h, encode_h.repeat(width, 1, 1))
        energy_w = torch.matmul(feat_w, encode_w.repeat(height, 1, 1))
        full_relation_h = self.softmax(energy_h)  # [b*w, c, c]
        full_relation_w = self.softmax(energy_w)

        full_aug_h = torch.bmm(full_relation_h, feat_h).view(batch_size, width, -1, height).permute(0, 2, 3, 1)
        full_aug_w = torch.bmm(full_relation_w, feat_w).view(batch_size, height, -1, width).permute(0, 2, 1, 3)
        out = self.gamma * (full_aug_h + full_aug_w) + x
        
        out = self.conv(out)
        
        return out
    
    
class MobileOne(nn.Module):
    """
    default mobileOne small model
    """

    def __init__(self, in_channels=3, num_classes=10, a = [0.75,0.75,1,1,1,2,2], k = [4,4,4,4,4,4]):

        super(MobileOne, self).__init__()

        ch = [int(x * y) for x,y in zip([64, 64, 128, 256, 256, 512], a)]

        self.block1 = MobileOneBlock(in_channels, ch[0], k[0], stride=2)

        self.block2 = nn.Sequential(
                MobileOneBlock(ch[0], ch[1], k[1], stride=2), 
                MobileOneBlock(ch[1], ch[1], k[1])
        )

        self.block3 = nn.Sequential(
                MobileOneBlock(ch[1], ch[2], k[2], stride=2), 
                *[MobileOneBlock(ch[2], ch[2], k[2]) for _ in range(7)]
        )

        self.block4 = nn.Sequential(
                MobileOneBlock(ch[2], ch[3], k[3], stride=2),
                *[MobileOneBlock(ch[3], ch[3], k[3]) for _ in range(4)]
        )
            
        self.block5 = nn.Sequential(
                MobileOneBlock(ch[3], ch[4], k[4], stride=2),
                *[MobileOneBlock(ch[4], ch[4], k[4]) for _ in range(4)]
        )
        
        
        self.block_decoder3 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),    # 64* scale_factor
                ConvRelu(ch[2], 32)
        )
        
        self.block_decoder4 = nn.Sequential(
                nn.Upsample(scale_factor=4, mode='bilinear'),    # 32* scale_factor
                ConvRelu(ch[3], 32)
        )
        
        self.block_decoder5 = nn.Sequential(
                nn.Upsample(scale_factor=8, mode='bilinear'),    # 16* scale_factor
                ConvRelu(ch[4], 32)
        )
        
        self.fca = FullyAttentionalBlock(32*3, 32)  # fully connection attention
        
        self.ConvRelu_last = ConvRelu(32, num_classes)
        
        self.out_sample = nn.Upsample(scale_factor=4, mode='bilinear')


    def switch_to_deploy(self):

        self.block1.switch_to_deploy()
        for b in self.block2:
            b.switch_to_deploy()
        for b in self.block3:
            b.switch_to_deploy()
        for b in self.block4:
            b.switch_to_deploy()
        for b in self.block5:
            b.switch_to_deploy()
        # self.block6.switch_to_deploy()


    def forward(self, x):
        _,_,h,w = x.size()
        
        # encoder layer
        x = self.block1(x)
        
        x2 = self.block2(x)

        x3 = self.block3(x2)

        x4 = self.block4(x3)

        x5 = self.block5(x4)
        
        # decoder layer 
        x3 = self.block_decoder3(x3)
        x4 = self.block_decoder4(x4)
        x5 = self.block_decoder5(x5)

        out = self.fca(torch.cat([x3, x4, x5], 1))
        out = self.ConvRelu_last(out)
        
        out = self.out_sample(out)
        
        return out
        
    
    
if __name__ == '__main__':
    from torch.autograd import Variable
    device = torch.device("cpu")
    
    input1 = torch.randn(2, 3, 512, 512)
    input1 = Variable(input1).to(device)
    model = MobileOne(in_channels=3, num_classes=2).to(device)
    
    output = model(input1)
    print(output.size())
    


    # print(output.size())
