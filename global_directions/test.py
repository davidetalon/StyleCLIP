import torch
import clip
from manipulate import Manipulator
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from MapTS import GetFs,GetBoundary,GetDt

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False ) 

M=Manipulator(dataset_name='ffhq') 
fs3=np.load('./npy/ffhq/fs3.npy')
np.set_printoptions(suppress=True)

img_index = 0
latents=torch.load('/home/davidetalon/Dev/encoder4editing/out_dir/test/latents.pt', map_location=torch.device('cpu'))
latents = latents[:3].cuda()
w_plus=latents.cpu().detach().numpy()
dlatents_loaded=M.W2S(w_plus)

img_index = 0
img_indexs=[img_index]

dlatent_tmp=[tmp[img_indexs] for tmp in dlatents_loaded]
M.num_images=len(img_indexs)

M.alpha=[0]
M.manipulate_layers=[0]
codes,out=M.EditOneC(0, dlatent_tmp) 
original=Image.fromarray(out[0,0]).resize((512,512))
M.manipulate_layers=None

neutral='face with eyes' #@param {type:"string"}
target='face with blue eyes' #@param {type:"string"}
classnames=[target,neutral]
dt=GetDt(classnames,model)

beta = 0.15 #@param {type:"slider", min:0.08, max:0.3, step:0.01}
alpha = 4.1 #@param {type:"slider", min:-10, max:10, step:0.1}
M.alpha=[alpha]
boundary_tmp2,c=GetBoundary(fs3,dt,M,threshold=beta)
codes=M.MSCode(dlatent_tmp,boundary_tmp2)
out=M.GenerateImg(codes)
generated=Image.fromarray(out[0,0])#.resize((512,512))


plt.figure(figsize=(20,7), dpi= 100)
plt.subplot(1,2,1)
plt.imshow(original)
plt.title('original')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(generated)
plt.title('manipulated')
plt.axis('off')