# test1
import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
import streamlit_jupyter as st_jupyter

# 初始化模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "stabilityai/stable-diffusion-2-1-base"
pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)

# 使用 streamlit_jupyter 运行
with st_jupyter.StreamlitRunner():
    st.title("Stable Diffusion 图片生成")

    # 用户输入提示
    prompt = st.text_input("输入提示词：", "Beautiful picture of a wave breaking")
    negative_prompt = st.text_input("输入负面提示词：", "zoomed in, blurry, oversaturated, warped")
    guidance_scale = st.slider("引导尺度 (guidance scale)", 1.0, 20.0, 8.0)
    num_inference_steps = st.slider("推理步数", 10, 50, 30)

    # 当点击按钮时，生成图片
    if st.button("生成图片"):
        with st.spinner("正在生成图片..."):
            # 编码提示词
            text_embeddings = pipe._encode_prompt(prompt, device, 1, True, negative_prompt)
            # 生成随机噪声
            latents = torch.randn((1, 4, 64, 64), device=device)
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
            pil_image = pipe.numpy_to_pil(image)[0]
            
            # 展示图片
            st.image(pil_image, caption="生成的图片", use_column_width=True)
