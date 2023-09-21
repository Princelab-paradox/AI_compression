import math
import io
import os
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_msssim import ms_ssim

# loading the models
from compressai.zoo import bmshj2018_factorized 
from compressai.zoo import bmshj2018_hyperprior 
from compressai.zoo import cheng2020_anchor

import gradio as gr

# function to compress the image 
def compress(input_img):
    # checking the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = bmshj2018_factorized(quality=1, pretrained=True).eval().to(device)

    print(f'Parameters: {sum(p.numel() for p in net.parameters())}')

    img = Image.open(input_img).convert('RGB')
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out_net = net.forward(x)
    out_net['x_hat'].clamp_(0, 1)
    # print(out_net.keys())

    rec_net = transforms.ToPILImage()(out_net['x_hat'].squeeze().cpu())
    
    # Split the file path into components
    components = input_img.split('/')

    # Get the last component (file name and extension)
    file_name_with_extension = components[-1]

    # Split the file name and extension using rsplit()
    file_name, file_extension = file_name_with_extension.rsplit('.', 1)

    rec_net.save("./result/"+file_name+".jpg")

    output_image = "./result/" + file_name + ".jpg"

    # calculatinig the reduction size
    file_size_bytes1 = os.path.getsize(input_img) 
    file_size_bytes2 = os.path.getsize(output_image)
    file_size_mb1 = file_size_bytes1 / 1000000
    file_size_mb2 = file_size_bytes2 / 1000000
    final_percent = ((file_size_mb1 - file_size_mb2) / file_size_mb1) * 100

    return output_image ,final_percent


# initialize the gradio UI

# image_input = gr.inputs.Image(label="Input Image")
# final_percentage = gr.outputs.Label(label="Reduction Percent")

# interface = gr.Interface(fn=compress, inputs=image_input, outputs=final_percentage)
# interface.launch()


# implementation of the AI image compression

input_img = "./images/dance.png"
output_image,final_percent = compress(input_img)
# print the output image path
print("The output image path::")
print("********************")
print(output_image)
# print the percentage 
print("********************")
print("The file is reduced by:: ")
print(final_percent)
print("********************")
    