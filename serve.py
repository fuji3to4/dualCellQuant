from fastapi import FastAPI
import gradio as gr
from dualCellQuant import build_ui

demo = build_ui().queue()
app = FastAPI() 
app = gr.mount_gradio_app(app, demo, path="/dualcellquant") 
