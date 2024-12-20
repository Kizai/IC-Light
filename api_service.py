import os
import torch
import numpy as np
import gc
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import io
import base64
from typing import Optional, List
import asyncio
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
import tempfile
import uuid
import gradio as gr
import threading
import logging
from logging.handlers import RotatingFileHandler
import time

# 导入必要的模型和处理函数
from main import (
    process_relight,
    BGSource,
    tokenizer,
    text_encoder,
    vae,
    unet,
    rmbg
)

# 配置日志
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    handlers=[RotatingFileHandler('logs/api.log', maxBytes=50*1024*1024, backupCount=5)],
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('api_service')

app = FastAPI(title="SceneMagic API", version="1.0.0")

# 添加请求日志中间件
@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(
        f"Method: {request.method} Path: {request.url.path} "
        f"Status: {response.status_code} "
        f"Process Time: {process_time:.2f}s"
    )
    return response

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定义请求模型
class RelightRequest(BaseModel):
    # 必填参数，没有默认值
    prompt: str
    image_width: int
    image_height: int
    num_samples: int = 1  # 默认生成1张
    
    # 可选参数，有默认值
    seed: int = 12345
    steps: int = 25
    a_prompt: str = "best quality"
    n_prompt: str = "lowres, bad anatomy, bad hands, cropped, worst quality"
    cfg: float = 2.0
    highres_scale: float = 1.5
    highres_denoise: float = 0.5
    lowres_denoise: float = 0.9
    bg_source: str = "None"

    # 添加参数验证
    class Config:
        @classmethod
        def schema_extra(cls, schema: dict) -> None:
            schema["example"] = {
                "prompt": "sunshine from window",
                "image_width": 800,
                "image_height": 800,
                "num_samples": 1,
                "seed": 12345,
                "bg_source": "None"
            }

    # 添加参数验证
    def validate_dimensions(self) -> "RelightRequest":
        if not (512 <= self.image_width <= 1600):
            raise ValueError("image_width must be between 512 and 1600")
        if not (512 <= self.image_height <= 1600):
            raise ValueError("image_height must be between 512 and 1600")
        if not (1 <= self.num_samples <= 12):
            raise ValueError("num_samples must be between 1 and 12")
        return self

# 添加新的请求模型，支持base64图片
class RelightRequestBase64(BaseModel):
    image_base64: str  # base64编码的图片
    prompt: str
    image_width: int
    image_height: int
    num_samples: int
    seed: int = 12345
    steps: int = 25
    a_prompt: str = "best quality, realistic colors, original materials and textures, consistent product appearance, vibrant and accurate colors, true-to-life details, no artistic filters, retain natural product surface, realistic lighting"
    n_prompt: str = "lowres, bad anatomy, bad hands, cropped, worst quality, unrealistic colors, faded colors, oversaturated, plastic textures, incorrect materials, artistic effects, bad lighting, blurry textures, distorted surfaces"
    cfg: float = 2.0
    highres_scale: float = 1.5
    highres_denoise: float = 0.5
    lowres_denoise: float = 0.9
    bg_source: str = "None"

    def validate_dimensions(self) -> "RelightRequestBase64":
        if not (512 <= self.image_width <= 1600):
            raise ValueError("image_width must be between 512 and 1600")
        if not (512 <= self.image_height <= 1600):
            raise ValueError("image_height must be between 512 and 1600")
        if not (1 <= self.num_samples <= 12):
            raise ValueError("num_samples must be between 1 and 12")
        return self

# 工具函数：将PIL Image转换为base64
def image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# 工具函数：base64转PIL Image
def base64_to_image(base64_str: str) -> Image.Image:
    try:
        # 如果base64字符串包含header（如：data:image/jpeg;base64,），去掉它
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        raise ValueError(f"Invalid base64 image: {str(e)}")

# 添加一个临时文件存储路径
TEMP_DIR = "./temp_downloads"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

@app.post("/relight")
async def relight(
    request: RelightRequest,
    file: UploadFile = File(...),
):
    try:
        logger.info(f"Received relight request - Prompt: {request.prompt}")
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # 验证参数
        request.validate_dimensions()
        
        # 读取和处理上传的图片
        contents = await file.read()
        input_image = Image.open(io.BytesIO(contents))
        input_array = np.array(input_image)

        # 在线程池中运行耗时操作
        preprocessed, results = await run_in_threadpool(
            process_relight,
            input_array,
            request.prompt,
            request.image_width,
            request.image_height,
            request.num_samples,
            request.seed,
            request.steps,
            request.a_prompt,
            request.n_prompt,
            request.cfg,
            request.highres_scale,
            request.highres_denoise,
            request.lowres_denoise,
            request.bg_source
        )

        # 转换结果为base64
        results_b64 = [image_to_base64(Image.fromarray(img)) for img in results]

        return {
            "status": "success",
            "prompt": request.prompt,
            "results": results_b64
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing relight request: {str(e)}")
        raise

@app.post("/relight/download")
async def relight_download(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    image_width: int = Form(800),
    image_height: int = Form(800),
    num_samples: int = Form(1),
    seed: int = Form(12345),
    steps: int = Form(25),
    a_prompt: str = Form("best quality"),
    n_prompt: str = Form("lowres, bad anatomy, bad hands, cropped, worst quality"),
    cfg: float = Form(2.0),
    highres_scale: float = Form(1.5),
    highres_denoise: float = Form(0.5),
    lowres_denoise: float = Form(0.9),
    bg_source: str = Form("None"),
):
    try:
        logger.info(f"Received relight download request - Prompt: {prompt}")
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # 验证参数
        request = RelightRequest(
            prompt=prompt,
            image_width=image_width,
            image_height=image_height,
            num_samples=num_samples,
            seed=seed,
            steps=steps,
            a_prompt=a_prompt,
            n_prompt=n_prompt,
            cfg=cfg,
            highres_scale=highres_scale,
            highres_denoise=highres_denoise,
            lowres_denoise=lowres_denoise,
            bg_source=bg_source
        )
        request.validate_dimensions()
        
        # 处理图片
        contents = await file.read()
        input_image = Image.open(io.BytesIO(contents))
        input_array = np.array(input_image)

        # 在线程池中运行耗时操作
        preprocessed, results = await run_in_threadpool(
            process_relight,
            input_array,
            request.prompt,
            request.image_width,
            request.image_height,
            request.num_samples,
            request.seed,
            request.steps,
            request.a_prompt,
            request.n_prompt,
            request.cfg,
            request.highres_scale,
            request.highres_denoise,
            request.lowres_denoise,
            request.bg_source
        )

        # 生成唯一文件名
        file_id = str(uuid.uuid4())
        
        # 保存为单张PNG图片
        filename = f"{file_id}.png"
        filepath = os.path.join(TEMP_DIR, filename)
        # 如果生成多张图片，只保存第一张
        Image.fromarray(results[0]).save(filepath)

        # 返回图片文件
        response = FileResponse(
            filepath,
            media_type='image/png',
            filename=f"result_{file_id}.png",
            headers={"X-Prompt": prompt}
        )
        
        # 创建清理任务
        asyncio.create_task(cleanup_file(filepath))
        
        return response

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing relight download request: {str(e)}")
        raise

@app.post("/relight/base64")
async def relight_base64(request: RelightRequestBase64):
    try:
        logger.info(f"Received relight base64 request - Prompt: {request.prompt}")
        # 验证参数
        request.validate_dimensions()
        
        # 解码base64图片
        input_image = base64_to_image(request.image_base64)
        input_array = np.array(input_image)

        # 在线程池中运行耗时操作
        preprocessed, results = await run_in_threadpool(
            process_relight,
            input_array,
            request.prompt,
            request.image_width,
            request.image_height,
            request.num_samples,
            request.seed,
            request.steps,
            request.a_prompt,
            request.n_prompt,
            request.cfg,
            request.highres_scale,
            request.highres_denoise,
            request.lowres_denoise,
            request.bg_source
        )

        # 生成唯一文件名
        file_id = str(uuid.uuid4())
        
        # 保存为单张PNG图片
        filename = f"{file_id}.png"
        filepath = os.path.join(TEMP_DIR, filename)
        # 如果生成多张图片，只保存第一张
        Image.fromarray(results[0]).save(filepath)

        # 返回图片文件
        response = FileResponse(
            filepath,
            media_type='image/png',
            filename=f"result_{file_id}.png",
            headers={"X-Prompt": request.prompt}
        )
        
        # 创建清理任务
        asyncio.create_task(cleanup_file(filepath))
        
        return response

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing relight base64 request: {str(e)}")
        raise

async def cleanup_file(filepath: str):
    """异步清理临文件"""
    await asyncio.sleep(300)  # 5分钟后删除
    if os.path.exists(filepath):
        os.remove(filepath)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

def run_gradio():
    """在单独的线程中运行Gradio"""
    block = gr.Blocks()
    # ... 这里放原来main.py中的Gradio界面代码 ...
    block.launch(server_name='0.0.0.0', server_port=7860)

if __name__ == "__main__":
    import uvicorn
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    uvicorn.run(app, host="0.0.0.0", port=8000) 
