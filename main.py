import warnings
import logging

# 过滤警告信息
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 设置日志级别
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import os
import math
import gradio as gr
import numpy as np
import torch
import safetensors.torch as sf
import db_examples
import json
from PIL import Image
from diffusers import (
    StableDiffusionPipeline, 
    StableDiffusionImg2ImgPipeline,
    AutoencoderKL, 
    UNet2DConditionModel, 
    DDIMScheduler,
    EulerAncestralDiscreteScheduler, 
    DPMSolverMultistepScheduler
)
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from briarmbg import BriaRMBG
from enum import Enum
from torch.hub import download_url_to_file
from huggingface_hub import hf_hub_download
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from functools import partial


# 'stablediffusionapi/realistic-vision-v51'
# 'runwayml/stable-diffusion-v1-5'
sd15_name = 'stablediffusionapi/realistic-vision-v51'
tokenizer = CLIPTokenizer.from_pretrained(sd15_name, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(sd15_name, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(sd15_name, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(sd15_name, subfolder="unet")
rmbg = BriaRMBG.from_pretrained("briaai/RMBG-1.4")

# Change UNet

with torch.no_grad():
    new_conv_in = torch.nn.Conv2d(8, unet.conv_in.out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding)
    new_conv_in.weight.zero_()
    new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
    new_conv_in.bias = unet.conv_in.bias
    unet.conv_in = new_conv_in

unet_original_forward = unet.forward


def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
    c_concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample)
    c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
    new_sample = torch.cat([sample, c_concat], dim=1)
    kwargs['cross_attention_kwargs'] = {}
    return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)


unet.forward = hooked_unet_forward

# Load

model_path = './models/iclight_sd15_fc.safetensors'

if not os.path.exists(model_path):
    model_path = hf_hub_download(
        repo_id="lllyasviel/ic-light",
        filename="iclight_sd15_fc.safetensors",
        cache_dir="./models"
    )

sd_offset = sf.load_file(model_path)
sd_origin = unet.state_dict()
keys = sd_origin.keys()
sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
unet.load_state_dict(sd_merged, strict=True)
del sd_offset, sd_origin, sd_merged, keys

# Device

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

device = get_device()
print(f"Using device: {device}")

# 修改数据类型逻辑
def get_dtype(device):
    if device.type == 'cuda':
        return torch.float16
    elif device.type == 'mps':
        return torch.float32  # MPS可能不支持某些半精度操作
    else:
        return torch.float32

dtype = get_dtype(device)
print(f"Using dtype: {dtype}")

# 更新模型加载部分
text_encoder = text_encoder.to(device=device, dtype=dtype)
vae = vae.to(device=device, dtype=dtype)
unet = unet.to(device=device, dtype=dtype)
rmbg = rmbg.to(device=device, dtype=torch.float32)

# SDP

unet.set_attn_processor(AttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())

# Samplers

ddim_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

euler_a_scheduler = EulerAncestralDiscreteScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    steps_offset=1
)

dpmpp_2m_sde_karras_scheduler = DPMSolverMultistepScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    algorithm_type="sde-dpmsolver++",
    use_karras_sigmas=True,
    steps_offset=1
)

# Pipelines

t2i_pipe = StableDiffusionPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=dpmpp_2m_sde_karras_scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
    image_encoder=None
)

i2i_pipe = StableDiffusionImg2ImgPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=dpmpp_2m_sde_karras_scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
    image_encoder=None
)


@torch.inference_mode()
def encode_prompt_inner(txt: str):
    # 添加长度限制
    if len(txt) > 512:  # 设置一个合理的最大长度
        txt = txt[:512]
        
    max_length = tokenizer.model_max_length
    chunk_length = tokenizer.model_max_length - 2
    id_start = tokenizer.bos_token_id
    id_end = tokenizer.eos_token_id
    id_pad = id_end

    def pad(x, p, i):
        return x[:i] if len(x) >= i else x + [p] * (i - len(x))

    tokens = tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
    chunks = [[id_start] + tokens[i: i + chunk_length] + [id_end] for i in range(0, len(tokens), chunk_length)]
    chunks = [pad(ck, id_pad, max_length) for ck in chunks]

    token_ids = torch.tensor(chunks).to(device=device, dtype=torch.int64)
    conds = text_encoder(token_ids).last_hidden_state

    return conds


@torch.inference_mode()
def encode_prompt_pair(positive_prompt, negative_prompt):
    c = encode_prompt_inner(positive_prompt)
    uc = encode_prompt_inner(negative_prompt)

    c_len = float(len(c))
    uc_len = float(len(uc))
    max_count = max(c_len, uc_len)
    c_repeat = int(math.ceil(max_count / c_len))
    uc_repeat = int(math.ceil(max_count / uc_len))
    max_chunk = max(len(c), len(uc))

    c = torch.cat([c] * c_repeat, dim=0)[:max_chunk]
    uc = torch.cat([uc] * uc_repeat, dim=0)[:max_chunk]

    c = torch.cat([p[None, ...] for p in c], dim=1)
    uc = torch.cat([p[None, ...] for p in uc], dim=1)

    return c, uc


@torch.inference_mode()
def pytorch2numpy(imgs, quant=True):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)

        if quant:
            y = y * 127.5 + 127.5
            y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        else:
            y = y * 0.5 + 0.5
            y = y.detach().float().cpu().numpy().clip(0, 1).astype(np.float32)

        results.append(y)
    return results


@torch.inference_mode()
def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0  # so that 127 must be strictly 0.0
    h = h.movedim(-1, 1)
    return h


def resize_and_center_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    original_width, original_height = pil_image.size
    scale_factor = max(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))
    resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)
    left = (resized_width - target_width) / 2
    top = (resized_height - target_height) / 2
    right = (resized_width + target_width) / 2
    bottom = (resized_height + target_height) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))
    return np.array(cropped_image)


def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)


@torch.inference_mode()
def run_rmbg(img, sigma=0.0):
    H, W, C = img.shape
    assert C == 3
    k = (256.0 / float(H * W)) ** 0.5
    feed = resize_without_crop(img, int(64 * round(W * k)), int(64 * round(H * k)))
    feed = numpy2pytorch([feed]).to(device=device, dtype=torch.float32)
    alpha = rmbg(feed)[0][0]
    alpha = torch.nn.functional.interpolate(alpha, size=(H, W), mode="bilinear")
    alpha = alpha.movedim(1, -1)[0]
    alpha = alpha.detach().float().cpu().numpy().clip(0, 1)
    result = 127 + (img.astype(np.float32) - 127 + sigma) * alpha
    return result.clip(0, 255).astype(np.uint8), alpha


@torch.inference_mode()
def process(input_fg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source):
    progress = gr.Progress()  # 创建一个进度对象
    
    bg_source = BGSource(bg_source)
    input_bg = None
    
    if bg_source == BGSource.NONE:
        pass
    elif bg_source == BGSource.LEFT:
        gradient = np.linspace(255, 0, image_width)
        image = np.tile(gradient, (image_height, 1))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.RIGHT:
        gradient = np.linspace(0, 255, image_width)
        image = np.tile(gradient, (image_height, 1))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.TOP:
        gradient = np.linspace(255, 0, image_height)[:, None]
        image = np.tile(gradient, (1, image_width))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.BOTTOM:
        gradient = np.linspace(0, 255, image_height)[:, None]
        image = np.tile(gradient, (1, image_width))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    else:
        raise 'Wrong initial latent!'

    rng = torch.Generator(device=device).manual_seed(int(seed))

    fg = resize_and_center_crop(input_fg, image_width, image_height)

    concat_conds = numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor

    conds, unconds = encode_prompt_pair(positive_prompt=prompt + ', ' + a_prompt, negative_prompt=n_prompt)

    if input_bg is None:
        progress(0.4)  # 只传入进度值
        
        latents = t2i_pipe(
            prompt_embeds=conds,
            negative_prompt_embeds=unconds,
            width=image_width,
            height=image_height,
            num_inference_steps=steps,
            num_images_per_prompt=num_samples,
            generator=rng,
            output_type='latent',
            guidance_scale=cfg,
            cross_attention_kwargs={'concat_conds': concat_conds},
        ).images.to(vae.dtype) / vae.config.scaling_factor
    else:
        bg = resize_and_center_crop(input_bg, image_width, image_height)
        bg_latent = numpy2pytorch([bg]).to(device=vae.device, dtype=vae.dtype)
        bg_latent = vae.encode(bg_latent).latent_dist.mode() * vae.config.scaling_factor
        progress(0.4)  # 只传入进度值
        
        latents = i2i_pipe(
            image=bg_latent,
            strength=lowres_denoise,
            prompt_embeds=conds,
            negative_prompt_embeds=unconds,
            width=image_width,
            height=image_height,
            num_inference_steps=int(round(steps / lowres_denoise)),
            num_images_per_prompt=num_samples,
            generator=rng,
            output_type='latent',
            guidance_scale=cfg,
            cross_attention_kwargs={'concat_conds': concat_conds},
        ).images.to(vae.dtype) / vae.config.scaling_factor

    progress(0.6)  # 更新进度
    
    pixels = vae.decode(latents).sample
    pixels = pytorch2numpy(pixels)
    pixels = [resize_without_crop(
        image=p,
        target_width=int(round(image_width * highres_scale / 64.0) * 64),
        target_height=int(round(image_height * highres_scale / 64.0) * 64))
    for p in pixels]

    progress(0.8)  # 更新进度
    
    pixels = numpy2pytorch(pixels).to(device=vae.device, dtype=vae.dtype)
    latents = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor
    latents = latents.to(device=unet.device, dtype=unet.dtype)

    image_height, image_width = latents.shape[2] * 8, latents.shape[3] * 8

    fg = resize_and_center_crop(input_fg, image_width, image_height)
    concat_conds = numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor

    progress(0.95)  # 更新进度
    
    latents = i2i_pipe(
        image=latents,
        strength=highres_denoise,
        prompt_embeds=conds,
        negative_prompt_embeds=unconds,
        width=image_width,
        height=image_height,
        num_inference_steps=int(round(steps / highres_denoise)),
        num_images_per_prompt=num_samples,
        generator=rng,
        output_type='latent',
        guidance_scale=cfg,
        cross_attention_kwargs={'concat_conds': concat_conds},
    ).images.to(vae.dtype) / vae.config.scaling_factor

    pixels = vae.decode(latents).sample

    return pytorch2numpy(pixels)


@torch.inference_mode()
def process_relight(input_fg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source):
    input_fg, matting = run_rmbg(input_fg)
    results = process(input_fg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source)
    return input_fg, results


quick_prompts = [
    'soft natural light, panoramic window, children\'s room',
    'pastel walls, whimsical cloud mural, minimalism',
    'light oak wood flooring, polished texture, modern child playroom',
    'cute and playful aesthetic, pastel color palette, child-friendly design',
    'round rug with animal pattern, reading nook with plush cushions',
    'fairy lights, small play tent, cozy atmosphere',
    'ultra-clean design, minimalist toy shelf, colorful toys and books',
    'whimsical room decor, modern minimalist aesthetics, photorealistic',
    'natural material textures, ambient lighting, cozy playroom',
    'cozy reading nook by the window, soft shadows and highlights',
    'open space in center, product placement ready, polished styling',
    'European minimalist children\'s room, cute yet modern aesthetic',
    'ultra-high-definition details, modern child-friendly environment',
    'fairy tale vibes, warmth and charm, designed for playful imagination',
    'cloud mural, pastel whimsy, Scandinavian-design inspired'
]
quick_prompts = [[x] for x in quick_prompts]


quick_subjects = [
    "Cozy European minimalist children's bedroom, soft peach-colored walls, smooth natural wood flooring",
    "Large window with sheer white curtains, bright natural light, airy and clean design",
    "Playful decor with pastel accents, colorful play kitchen, animal-themed rug, modern wall shelves filled with toys and books",
    "Scandinavian design, ultra-clean and organized, child-friendly furniture, cinematic lighting, photorealistic textures",
    "Cozy reading nook with large cushion by the window, spacious central area for play or product placement"
]
quick_subjects = [[x] for x in quick_subjects]


class BGSource(Enum):
    NONE = "None"
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"


# 添加用户验证函数
def check_login(username, password):
    try:
        with open('users.json', 'r') as f:
            config = json.load(f)
            # 获取用户列表
            users = config.get('users', {})
            # 验证用户名和密码
            if username in users and users[username] == password:
                return True
    except Exception as e:
        print(f"读取用户配置失败: {str(e)}")
    return False


# 简化登录函数
def login(username, password):
    if check_login(username, password):
        return {
            "message": "登录成功！",
            "visible": True
        }
    return {
        "message": "用户名或密码错误！",
        "visible": False
    }

# 建FastAPI应用
app = FastAPI(title="SceneMagic API", version="1.0.0")

# ... FastAPI相关代码 ...

# 添加SessionManager类定义
class SessionManager:
    def __init__(self, session_dir="./sessions"):
        self.session_dir = session_dir
        if not os.path.exists(session_dir):
            os.makedirs(session_dir)

    def save_session(self, username, password):
        session_id = str(hash(f"{username}_{time.time()}"))
        session_data = {
            "username": username,
            "password": password,
            "timestamp": time.time()
        }
        with open(os.path.join(self.session_dir, f"{session_id}.json"), "w") as f:
            json.dump(session_data, f)
        return session_id

    def load_session(self, session_id):
        try:
            session_file = os.path.join(self.session_dir, f"{session_id}.json")
            if os.path.exists(session_file):
                with open(session_file, "r") as f:
                    session_data = json.load(f)
                # 检查session是否过期（24小时）
                if time.time() - session_data["timestamp"] < 24 * 3600:
                    return session_data
                os.remove(session_file)
        except Exception as e:
            print(f"Session load error: {e}")
        return None

    def clear_session(self, session_id):
        try:
            session_file = os.path.join(self.session_dir, f"{session_id}.json")
            if os.path.exists(session_file):
                os.remove(session_file)
        except Exception as e:
            print(f"Session clear error: {e}")

# 创建session管理器
session_manager = SessionManager()


# 创建Gradio界面
block = gr.Blocks()

def check_login_required(fn):
    """登录检查装饰器"""
    def wrapper(*args, **kwargs):
        # ��取最后一个参数作为登录状态
        is_logged_in = args[-1]
        if not is_logged_in:
            raise gr.Error("请先登录后使用")
        # 移除is_logged_in参数再传递给原函数
        args = args[:-1]
        return fn(*args)
    return wrapper

@check_login_required
def process_relight_with_auth(*args):
    try:
        progress = gr.Progress()
        progress(0)  # 初始进度
        
        # 预处理
        input_fg, matting = run_rmbg(args[0])
        progress(0.2)  # 更新进度
        
        # 生成过程
        results = process(input_fg, *args[1:])
        progress(1.0)  # 最终进度
        
        return input_fg, results
    except Exception as e:
        raise gr.Error(f"生成失败: {str(e)}")

with block:
    # 只保留登录状态
    is_logged_in = gr.State(False)
    
    # 登录界面
    with gr.Column(elem_id="login_interface") as login_interface:
        with gr.Row():
            gr.Markdown("## 登录")
        
        with gr.Row():
            with gr.Column():
                username = gr.Textbox(label="用户名")
                password = gr.Textbox(label="密码", type="password")
                login_button = gr.Button("登录")
                login_status = gr.Markdown("请登录后使用")

    # 主界面
    with gr.Column(elem_id="main_interface", visible=False) as main_interface:
        with gr.Row():
            with gr.Column(scale=11):
                gr.Markdown("## SceneMagic-场景魔法")

        with gr.Row():
            with gr.Column():
                with gr.Row():
                    input_fg = gr.Image(source='upload', type="numpy", label="Image", height=480)
                    output_bg = gr.Image(type="numpy", label="Preprocessed Foreground", height=480)
                prompt = gr.Textbox(label="Prompt")
                bg_source = gr.Radio(choices=[e.value for e in BGSource],
                                     value=BGSource.NONE.value,
                                     label="Lighting Preference (Initial Latent)", type='value')
                example_quick_subjects = gr.Dataset(samples=quick_subjects, label='Subject Quick List', samples_per_page=1000, components=[prompt])
                example_quick_prompts = gr.Dataset(samples=quick_prompts, label='Lighting Quick List', samples_per_page=1000, components=[prompt])
                relight_button = gr.Button(value="立即生成")

                with gr.Group():
                    with gr.Row():
                        num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                        seed = gr.Number(label="Seed", value=12345, precision=0)

                    with gr.Row():
                        image_width = gr.Slider(
                            label="Image Width", 
                            minimum=512,   # 修改最小值
                            maximum=1600,  # 修改最大值
                            value=1152,    # 修改默认值
                            step=48       # 修改步长为64
                        )
                        image_height = gr.Slider(
                            label="Image Height", 
                            minimum=512,   # 修改最小值
                            maximum=1600,  # 改最大值
                            value=1152,    # 改默认值
                            step=48       # 修改步长为64
                        )

                with gr.Accordion("Advanced options", open=False):
                    steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1)
                    cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=2, step=0.01)
                    lowres_denoise = gr.Slider(label="Lowres Denoise (for initial latent)", minimum=0.1, maximum=1.0, value=0.9, step=0.01)
                    highres_scale = gr.Slider(label="Highres Scale", minimum=1.0, maximum=3.0, value=1.5, step=0.01)
                    highres_denoise = gr.Slider(label="Highres Denoise", minimum=0.1, maximum=1.0, value=0.5, step=0.01)
                    a_prompt = gr.Textbox(label="Added Prompt", value='best quality, realistic colors, original materials and textures, consistent product appearance, vibrant and accurate colors, true-to-life details, no artistic filters, retain natural product surface, realistic lighting')
                    n_prompt = gr.Textbox(label="Negative Prompt", value='lowres, bad anatomy, bad hands, cropped, worst quality, unrealistic colors, faded colors, oversaturated, plastic textures, incorrect materials, artistic effects, bad lighting, blurry textures, distorted surfaces')
            with gr.Column():
                result_gallery = gr.Gallery(height=832, object_fit='contain', label='Outputs')

        # 修改图片处理事件，确保登录状态是最后一个参数
        ips = [
            input_fg, prompt, image_width, image_height, num_samples, 
            seed, steps, a_prompt, n_prompt, cfg, highres_scale, 
            highres_denoise, lowres_denoise, bg_source, is_logged_in  # is_logged_in放在最后
        ]

        # 修改生成按钮事件，恢复原来的配置
        relight_button.click(
            fn=process_relight_with_auth,
            inputs=ips,
            outputs=[output_bg, result_gallery],
            show_progress=True
        )
        example_quick_prompts.click(
            lambda x, y: ', '.join(y.split(', ')[:2] + [x[0]]), 
            inputs=[example_quick_prompts, prompt], 
            outputs=prompt, 
            show_progress=False
        )
        example_quick_subjects.click(
            lambda x: x[0], 
            inputs=example_quick_subjects, 
            outputs=prompt, 
            show_progress=False
        )

    def handle_login(username, password):
        result = login(username, password)
        if result["visible"]:  # 使用 visible 替代 token 检查
            return [
                result["message"],
                gr.Column(visible=False),  # 隐藏登录界面
                gr.Column(visible=True),   # 显示主界面
                True  # 只需要更新登录状态
            ]
        return [
            result["message"],
            gr.Column(visible=True),
            gr.Column(visible=False),
            False
        ]

    # 登录按钮事件
    login_button.click(
        fn=handle_login,
        inputs=[username, password],
        outputs=[
            login_status,
            login_interface,
            main_interface,
            is_logged_in
        ]
    )

# 将FastAPI挂载到Gradio
app = gr.mount_gradio_app(app, block, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
