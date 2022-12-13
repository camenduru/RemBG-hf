## Modified from Akhaliq Hugging Face Demo
## https://huggingface.co/akhaliq

import gradio as gr
import os
import cv2

def inference(file, mask, af):
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
    [
        gr.inputs.Image(type="filepath", label="Input"),
        gr.inputs.Slider(10, 25, default=10, label="Alpha matting"), 
        gr.inputs.Radio(
            [
                "Default", 
                "Mask only"
            ], 
            type="value",
            default="Alpha matting",
            label="Choices"
        )
    ], 
    gr.outputs.Image(type="file", label="Output"),
    title=title,
    description=description,
    article=article,
    examples=[["lion.png", 10, "Default"], ["girl.jpg", 10, "Default"]],
    enable_queue=True
    ).launch()