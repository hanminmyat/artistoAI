import gradio as gr
import torch
import numpy as np
from PIL import Image

model2 = torch.hub.load(
    "bryandlee/animegan2-pytorch:main", "generator", pretrained="face_paint_512_v2")
model1 = torch.hub.load("bryandlee/animegan2-pytorch:main",
                        "generator", pretrained="paprika")
face2paint = torch.hub.load(
    "bryandlee/animegan2-pytorch:main", "face2paint", size=720)

def inference(img, ver):
    # Convert the input image to a NumPy array
    img = np.array(img)
    original_size = img.shape[:2]  # Store the original height and width

    # Convert the input image to a torch.Tensor
    img_tensor = torch.tensor(img).permute(2, 0, 1).float() / 255.0

    if ver == 'Portrait':
        out = face2paint(model2, Image.fromarray(img))
    else:
        out = face2paint(model1, Image.fromarray(img))

    # Convert the output PIL Image to a NumPy array and resize to original size
    out = np.array(out)
    out = out.astype(np.uint8)

    # Resize the output back to the original size using PIL
    out = Image.fromarray(out)
    out = out.resize(original_size[::-1], Image.BILINEAR)

    return np.array(out)

# Define Gradio components using the latest API
image_input = gr.inputs.Image(type="pil")
radio_input = gr.inputs.Radio(['Landscape', 'Portrait'], default='Portrait', label='version')
image_output = gr.outputs.Image(type="pil")

# Create Gradio interface
demo = gr.Interface(
    fn=inference,
    inputs=[image_input, radio_input],
    outputs=image_output,
    theme="soft",
    allow_flagging=False,
    css="footer {visibility: hidden}"
)

demo.launch(share=True)