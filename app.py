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
def image_compress(input_img):
    # removing the file from folder

    folder_path = "./result/"
    file_name = "compressed.jpg"

    file_path = os.path.join(folder_path, file_name)
    # checking the compressed file exist or not
    if file_path:
        try:
            os.remove(file_path)
            print(f"File {file_name} deleted successfully.")
        except OSError as e:
            # print(f"Error: {file_path} - {e.strerror}")
            pass
   
    # checking the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = bmshj2018_factorized(quality=1, pretrained=True).eval().to(device)

    print(f'Parameters: {sum(p.numel() for p in net.parameters())}')

    # img = Image.open(input_img).convert('RGB')
    img = input_img.convert('RGB') 
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out_net = net.forward(x)
    out_net['x_hat'].clamp_(0, 1)

    rec_net = transforms.ToPILImage()(out_net['x_hat'].squeeze().cpu())

    rec_net.save("./result/compressed.jpg")

    output_image = "./result/compressed.jpg"

    # print("Your input image path is:::")
    # print(input_img.name)
    
    # Split the file path into components
    # components = input_img.split('/')

    # Get the last component (file name and extension)
    # file_name_with_extension = components[-1]

    # Split the file name and extension using rsplit()
    # file_name, file_extension = file_name_with_extension.rsplit('.', 1)

    # rec_net.save("./result/"+file_name+".jpg")

    # output_image = "./result/" + file_name + ".jpg"

    # calculatinig the reduction size
    # file_size_bytes1 = os.path.getsize(input_img)
    # file_size_bytes2 = os.path.getsize(output_image)
    # file_size_mb1 = file_size_bytes1 / 1000000
    # file_size_mb2 = file_size_bytes2 / 1000000
    # final_percent = ((file_size_mb1 - file_size_mb2) / file_size_mb1) * 100

    return output_image, output_image

# defining the components the inference
input_component = gr.Image(type="pil")
output_component = [gr.Image(type="pil"), gr.File(label="Download", extension=".png")]

interface = gr.Interface(
    fn=image_compress,
    inputs=input_component,
    outputs=output_component
)

interface.launch()