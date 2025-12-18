import gradio as gr
import torch
from diffusers import AudioLDMPipeline

# 1. 设置设备：如果有 GPU 则使用 GPU，否则使用 CPU
# 注意：在免费的 HuggingFace Space 上通常是 CPU，速度较慢；付费 Space 可用 GPU。
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 2. 加载模型 (首次运行会自动下载模型)
repo_id = "cvssp/audioldm-s-full-v2" # 使用 v2 版本，效果更好
pipe = AudioLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float32)
pipe = pipe.to(device)

# 3. 定义生成函数
def generate_sfx(prompt, duration, guidance_scale, n_candidates):
    # 稍微修饰一下提示词，优化音效质量
    full_prompt = f"{prompt}, high quality, high fidelity, clear sound"
    
    # 生成音频
    audio = pipe(
        prompt=full_prompt,
        num_inference_steps=20,     # 步数，越高质量越好但越慢
        audio_length_in_s=duration, # 时长
        guidance_scale=guidance_scale,
        num_waveforms_per_prompt=n_candidates
    ).audios[0]

    # 返回采样率和音频数据
    return (16000, audio)

# 4. 搭建网页界面 (UI)
with gr.Blocks(title="🎮 游戏音效生成器") as demo:
    gr.Markdown("# 🎮 AI 游戏音效生成器")
    gr.Markdown("输入提示词（英文），生成属于你的游戏素材！")
    
    with gr.Row():
        with gr.Column():
            # 输入控件
            text_input = gr.Textbox(label="提示词 (Prompt)", placeholder="例如: Laser gun shot, heavy sci-fi weapon")
            duration_slider = gr.Slider(minimum=1.0, maximum=10.0, value=5.0, step=0.5, label="时长 (秒)")
            guidance_slider = gr.Slider(minimum=0, maximum=5, value=2.5, step=0.5, label="提示词相关度 (Guidance Scale)")
            submit_btn = gr.Button("🚀 生成音效", variant="primary")
            
        with gr.Column():
            # 输出控件
            audio_output = gr.Audio(label="生成的音效", type="numpy")
    
    # 绑定点击事件
    submit_btn.click(
        fn=generate_sfx, 
        inputs=[text_input, duration_slider, guidance_slider, gr.Number(value=1, visible=False)], 
        outputs=audio_output
    )

    # 预设一些游戏常用提示词示例
    gr.Examples(
        examples=[
            ["Laser gun shot, sci-fi, pew pew sound"],
            ["Heavy stone door opening in a dungeon"],
            ["Collecting a gold coin, retro game style"],
            ["Footsteps on gravel, slow walking"],
            ["Magic spell casting, sparkles, chime"]
        ],
        inputs=text_input
    )

# 启动应用
if __name__ == "__main__":
    demo.launch()
