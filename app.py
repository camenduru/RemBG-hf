## Modified from Akhaliq Hugging Face Demo
## https://huggingface.co/akhaliq

import gradio as gr
import os
import cv2

def inference(file, af, mask):
    im = cv2.imread(file, cv2.IMREAD_COLOR)
    cv2.imwrite(os.path.join("input.png"), im)

    from rembg import remove

    input_path = 'input.png'
    output_path = 'output.png'

    with open(input_path, 'rb') as i:
        with open(output_path, 'wb') as o:
            input = i.read()
            output = remove(input, alpha_matting_erode_size = af, only_mask = (True if mask == "Mask only" else False))
            o.write(output)
    return os.path.join("output.png")
        
title = "RemBG"
description = "Gradio demo for RemBG. To use it, simply upload your image and wait. Read more at the link below."
article = "<p style='text-align: center;'><a href='https://github.com/danielgatis/rembg' target='_blank'>Github Repo</a></p>"


gr.Interface(
    inference, 
    [gr.inputs.Image(type="filepath", label="Input"), gr.Slider(10, 25, value=10, label="Alpha matting"), gr.Radio(choices = ["Alpha matting", "Mask only"], value = "Alpha matting")], 
    gr.outputs.Image(type="file", label="Output"),
    title=title,
    description=description,
    article=article,
    examples=[['lion.png']],
    enable_queue=True
    ).launch()