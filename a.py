import streamlit as st
import torch
from mini_stable_diffusion import MiniStableDiffusionPipeline  # 替换为实际库和类

# 初始化轻量级模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "path_to_ministable_diffusion_model"  # 替换为实际的 MiniStableDiffusion 模型路径或ID
pipe = MiniStableDiffusionPipeline.from_pretrained(model_id).to(device)

# 定义Streamlit应用
st.title("MiniStableDiffusion 图片生成")

# 用户输入提示
prompt = st.text_input("输入提示词：", "A serene landscape")
negative_prompt = st.text_input("输入负面提示词：", "blurry, distorted")
guidance_scale = st.slider("引导尺度 (guidance scale)", 1.0, 20.0, 5.0)  # 调整为适合轻量级模型的范围
num_inference_steps = st.slider("推理步数", 5, 20, 10)  # 减少步数以适应轻量级模型

# 当点击按钮时，生成图片
if st.button("生成图片"):
    with st.spinner("正在生成图片..."):
        # 编码提示词
        text_embeddings = pipe.encode_prompt(prompt, device, negative_prompt)  # 替换为实际的编码函数
        # 生成随机噪声
        latents = torch.randn((1, 4, 32, 32), device=device)  # 使用适合 MiniStableDiffusion 的尺寸
        latents *= pipe.scheduler.init_noise_sigma
        # 设置时间步长
        pipe.scheduler.set_timesteps(num_inference_steps, device=device)

        # 开始生成图片
        for i, t in enumerate(pipe.scheduler.timesteps):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
            with torch.no_grad():
                noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
        
        # 解码潜变量
        with torch.no_grad():
            image = pipe.decode_latents(latents.detach())
        
        # 将 numpy 数组转换为 PIL 图像
        pil_image = Image.fromarray(image.numpy()[0])
        
        # 展示图片
        st.image(pil_image, caption="生成的图片", use_column_width=True)
