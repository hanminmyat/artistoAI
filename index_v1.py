import gradio as gr
import torch

model2 = model = torch.hub.load(
    "bryandlee/animegan2-pytorch:main", "generator", pretrained="face_paint_512_v2")
model1 = torch.hub.load("bryandlee/animegan2-pytorch:main",
                        "generator", pretrained="paprika")
face2paint = torch.hub.load(
    "bryandlee/animegan2-pytorch:main", "face2paint", size=512)


def inference(img, ver):
    if ver == 'Portrait':
        out = face2paint(model2, img)
    else:
        out = face2paint(model1, img)
    return out


demo = gr.Interface(
    fn=inference,
    inputs=[gr.Image(type="pil"), gr.Radio(['Landscape',
                                            'Portrait'], type="value", default='Portrait', label='version')],
    outputs=gr.Image(type="pil"),
    theme=gr.themes.Soft(),
    allow_flagging="never",
    css="footer {visibility: hidden}"
)

demo.launch(share=True)