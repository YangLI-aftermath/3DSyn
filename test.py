import sys

import numpy as np

sys.path.append('/Users/menglidaren/Desktop/repo/3DSyn/styleganv2')
import torch
import styleganv2.training.depthgan as dp

# G_rgb Done!
'''
G_rgb = dp.GeneratorRGB(z_dim=512,c_dim=0,w_dim=512,img_resolution=128,img_channels=3)
channels_dict = { res: min(32768 // res, 512) for res in [4,8,16,32,64,128]}
print(channels_dict)
priors_first = [(None,torch.randn(3,channels_dict[4],4,4))]
priors = [(torch.randn(3,channels_dict[res//2],res//2,res//2), torch.randn(3,channels_dict[res],res,res)) for res in [8,16,32,64,128]]
priors = priors_first + priors
z = torch.randn(3,512)
print(G_rgb(z,None,priors).size())
'''

# G_d done!
'''
G_d = dp.GeneratorDepth(z_dim=512,c_dim=0,w_dim=512,img_resolution=128,img_channels=1,frequency=256)
z = torch.randn(3,512)
print(G_d(z,None,30).size())
'''

# SwitchD Done!
## Score
'''
D = dp.SwitchDiscriminator(c_dim=0,img_resolution=256,img_channels=3)
img = torch.randn(3,4,256,256)
print(D(img,None,'score').size())
'''
## Depth Prediction
'''
D = dp.SwitchDiscriminator(c_dim=0,img_resolution=256,img_channels=3)
img = torch.randn(3,3,256,256)
print(D(img,None,'depth').size())
'''