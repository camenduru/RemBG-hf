## Modified from Akhaliq Hugging Face Demo
## https://huggingface.co/akhaliq

import gradio as gr
import os
import cv2

def inference(file, af, mask, model):
    im = cv2.imread(file, cv2.IMREAD_COLOR)
    cv2.imwrite(os.path.join("input.png"), im)

    from rembg import remove
    from rembg.session_base import BaseSession
    from rembg.session_factory import new_session

    input_path = 'input.png'
    output_path = 'output.png'

    with open(input_path, 'rb') as i:
        with open(output_path, 'wb') as o:
            input = i.read()
            sessions: dict[str, BaseSession] = {}
            output = remove(
                input, 
                session=sessions.setdefault(
                    model, new_session(model)
                ),
                alpha_matting_erode_size = af, 
                only_mask = (True if mask == "Mask only" else False)
            )
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
            default="Default",
            label="Choices"
        ),
        gr.inputs.Dropdown([
            "u2net", 
            "u2netp", 
            "u2net_human_seg", 
            "u2net_cloth_seg", 
            "silueta"
            ], 
            type="value",
            default="u2net",
            label="Models"
        ),
    ], 
    gr.outputs.Image(type="file", label="Output"),
    title=title,
    description=description,
    article=article,
    examples=[["lion.png", 10, "Default", "u2net"], ["girl.jpg", 10, "Default", "u2net"]],
    enable_queue=True
    ).launch()