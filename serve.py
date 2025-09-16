from fastapi import FastAPI
import gradio as gr
from dualCellQuant import build_ui

demo = build_ui().queue()

# 重要: 外側のASGI(app)に root_path を付ける
app = FastAPI(root_path="/dualcellquant")

# 重要: Gradio は "/" にマウント（外側の root_path が /dualcellquant を吸収）
app = gr.mount_gradio_app(app, demo, path="/")
