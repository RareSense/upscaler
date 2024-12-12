import modal
from models import UpscaleRequest, UpscaleResponse
from fastapi import FastAPI, HTTPException, Request
from pathlib import Path
from fastapi import FastAPI
from models_init import SUPIR_device
from models import UpscaleRequest, UpscaleResponse
from fastapi import HTTPException
import time
from utils import base64_to_image, image_to_base64
from SUPIR.util import PIL2Tensor,Tensor2PIL
from config import args,logger

# CUDA version and dependencies
cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

# Define the Modal image with dependencies
upscaler_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install(
        "git",
        "libglib2.0-0",
        "libsm6",
        "libxrender1",
        "libxext6",
        "ffmpeg",
        "libgl1",
    )
    .pip_install(
        "accelerate==0.32.1",
        "aiofiles==23.2.1",
        "aiohappyeyeballs==2.4.4",
        "aiohttp==3.11.10",
        "aiosignal==1.3.1",
        "altair==5.5.0",
        "annotated-types==0.7.0",
        "antlr4-python3-runtime==4.9.3",
        "anyio==4.7.0",
        "async-timeout==5.0.1",
        "attrs==24.2.0",
        "braceexpand==0.1.7",
        "certifi==2024.8.30",
        "charset-normalizer==3.4.0",
        "clean-fid==0.1.35",
        "click==8.1.7",
        "clip-anytorch==2.6.0",
        "contourpy==1.3.1",
        "cycler==0.12.1",
        "dctorch==0.1.2",
        "diffusers==0.29.2",
        "dnspython==2.7.0",
        "docker-pycreds==0.4.0",
        "einops==0.8.0",
        "einops-exts==0.0.4",
        "email_validator==2.2.0",
        "exceptiongroup==1.2.2",
        "facexlib==0.3.0",
        "fastapi==0.111.1",
        "fastapi-cli==0.0.6",
        "ffmpy==0.4.0",
        "filelock==3.16.1",
        "filterpy==1.4.5",
        "fonttools==4.55.2",
        "frozenlist==1.5.0",
        "fsspec==2024.6.1",
        "ftfy==6.3.1",
        "gitdb==4.0.11",
        "GitPython==3.1.43",
        "grpclib==0.4.7",
        "h11==0.14.0",
        "h2==4.1.0",
        "hpack==4.0.0",
        "httpcore==1.0.7",
        "httptools==0.6.4",
        "httpx==0.27.0",
        "huggingface-hub==0.26.5",
        "hyperframe==6.0.1",
        "idna==3.10",
        "imageio==2.36.1",
        "importlib_metadata==8.5.0",
        "importlib_resources==6.4.5",
        "Jinja2==3.1.4",
        "joblib==1.4.2",
        "jsonmerge==1.9.2",
        "jsonschema==4.23.0",
        "jsonschema-specifications==2024.10.1",
        "k-diffusion==0.1.1.post1",
        "kiwisolver==1.4.7",
        "kornia==0.7.3",
        "kornia_rs==0.1.7",
        "lazy_loader==0.4",
        "lightning-utilities==0.11.9",
        "llvmlite==0.43.0",
        "Markdown==3.6",
        "markdown-it-py==3.0.0",
        "MarkupSafe==2.1.5",
        "matplotlib==3.9.1",
        "mdurl==0.1.2",
        "modal==0.68.2",
        "mpmath==1.3.0",
        "multidict==6.1.0",
        "narwhals==1.16.0",
        "networkx==3.4.2",
        "ninja==1.11.1.1",
        "numba==0.60.0",
        "numpy==1.26.4",
        "nvidia-cublas-cu12==12.1.3.1",
        "nvidia-cuda-cupti-cu12==12.1.105",
        "nvidia-cuda-nvrtc-cu12==12.1.105",
        "nvidia-cuda-runtime-cu12==12.1.105",
        "nvidia-cudnn-cu12==8.9.2.26",
        "nvidia-cufft-cu12==11.0.2.54",
        "nvidia-curand-cu12==10.3.2.106",
        "nvidia-cusolver-cu12==11.4.5.107",
        "nvidia-cusparse-cu12==12.1.0.106",
        "nvidia-nccl-cu12==2.20.5",
        "nvidia-nvjitlink-cu12==12.6.85",
        "nvidia-nvtx-cu12==12.1.105",
        "omegaconf==2.3.0",
        "open-clip-torch==2.24.0",
        "openai-clip==1.0.1",
        "opencv-python==4.10.0.84",
        "orjson==3.10.12",
        "packaging==24.2",
        "pandas==2.2.2",
        "pillow==10.4.0",
        "pillow_heif==0.18.0",
        "platformdirs==4.3.6",
        "propcache==0.2.1",
        "protobuf==5.29.1",
        "psutil==5.9.8",
        "pydantic==2.10.3",
        "pydantic_core==2.27.1",
        "pydub==0.25.1",
        "Pygments==2.18.0",
        "pyparsing==3.2.0",
        "python-dateutil==2.9.0.post0",
        "python-dotenv==1.0.1",
        "python-multipart==0.0.19",
        "pytorch-lightning==2.3.3",
        "pytz==2024.2",
        "PyYAML==6.0.1",
        "referencing==0.35.1",
        "regex==2024.11.6",
        "requests==2.32.3",
        "rich==13.9.4",
        "rich-toolkit==0.12.0",
        "rpds-py==0.22.3",
        "ruff==0.8.2",
        "safetensors==0.4.5",
        "scikit-image==0.24.0",
        "scikit-learn==1.5.1",
        "scipy==1.14.0",
        "semantic-version==2.10.0",
        "sentencepiece==0.2.0",
        "sentry-sdk==2.19.2",
        "setproctitle==1.3.4",
        "shellingham==1.5.4",
        "sigtools==4.0.1",
        "six==1.17.0",
        "smmap==5.0.1",
        "sniffio==1.3.1",
        "spaces==0.30.4",
        "starlette==0.37.2",
        "sympy==1.13.3",
        "synchronicity==0.9.6",
        "threadpoolctl==3.5.0",
        "tifffile==2024.9.20",
        "timm==1.0.7",
        "tokenizers==0.19.1",
        "toml==0.10.2",
        "tomlkit==0.12.0",
        "torch==2.3.1",
        "torchdiffeq==0.2.5",
        "torchmetrics==1.6.0",
        "torchsde==0.2.6",
        "torchvision==0.18.1",
        "tqdm==4.66.4",
        "trampoline==0.1.2",
        "transformers==4.42.4",
        "triton==2.3.1",
        "typer==0.15.1",
        "types-certifi==2021.10.8.3",
        "types-toml==0.10.8.20240310",
        "typing_extensions==4.12.2",
        "tzdata==2024.2",
        "urllib3==2.2.2",
        "uvicorn==0.30.1",
        "uvloop==0.21.0",
        "wandb==0.17.4",
        "watchfiles==1.0.0",
        "wcwidth==0.2.13",
        "webdataset==0.2.86",
        "websockets==11.0.3",
        "xformers==0.0.27",
        "yarl==1.18.3",
        "zipp==3.21.0",
    )
)

app = modal.App("supir_upscaler", image=upscaler_image)

# Initialize FastAPI app
web_app = FastAPI()

# modal.Volume.from_name("root", create_if_missing=True)
# vol = modal.Volume.lookup("root")

# with vol.batch_upload() as batch:
#     batch.put_directory("/home/nimra/sahal/SUPIR/SUPIR", "/root/SUPIR")
#     batch.put_directory("/home/nimra/sahal/SUPIR/agm", "/root/agm")

# Explicitly define paths
local_config_path = Path("/home/nimra/sahal/SUPIR/config.py").resolve()
local_utils_path = Path("/home/nimra/sahal/SUPIR/utils.py").resolve()
local_models_init_path = Path("/home/nimra/sahal/SUPIR/models_init.py").resolve()
local_models_path = Path("/home/nimra/sahal/SUPIR/models.py").resolve()
local_CKPT_PTH_path = Path("/home/nimra/sahal/SUPIR/CKPT_PTH.py").resolve()
local_SUPIR_v0_yaml_path = Path("/home/nimra/sahal/SUPIR/SUPIR_v0.yaml").resolve()
# local_SUPIR_path = Path("/home/nimra/sahal/SUPIR/SUPIR").resolve()
# local_sgm_path = Path("/home/nimra/sahal/SUPIR/sgm").resolve()

# Remote paths in the container
remote_config_path = Path("/root/config.py")
remote_utils_path = Path("/root/utils.py")
remote_models_init_path = Path("/root/models_init.py")
remote_models_path = Path("/root/models.py")
remote_CKPT_PTH_path = Path("/root/CKPT_PTH.py")
remote_SUPIR_v0_yaml_path = Path("/root/SUPIR_v0.yaml")
# remote_SUPIR_path = Path("/root/SUPIR")
# remote_sgm_path = Path("/root/sgm")

# Mount files to the container
mounts = [
    modal.Mount.from_local_file(local_models_init_path, remote_models_init_path),
    modal.Mount.from_local_file(local_utils_path, remote_utils_path),
    modal.Mount.from_local_file(local_config_path, remote_config_path),
    modal.Mount.from_local_file(local_models_path, remote_models_path),
    modal.Mount.from_local_file(local_CKPT_PTH_path, remote_CKPT_PTH_path),
    modal.Mount.from_local_file(local_SUPIR_v0_yaml_path, remote_SUPIR_v0_yaml_path),
    # modal.Mount.from_local_file(local_SUPIR_path, remote_SUPIR_path),
    # modal.Mount.from_local_file(local_sgm_path, remote_sgm_path),
]

@app.cls(
    gpu="A100",
    concurrency_limit=5,
    mounts=mounts,
    container_idle_timeout=120,  # in seconds
    volumes={
        "/root/laion_CLIP-ViT-bigG-14-laion2B-39B-b160k": modal.Volume.from_name("laion_CLIP-ViT-bigG-14-laion2B-39B-b160k", create_if_missing=True),  
        "/root/yushan777_SUPIR": modal.Volume.from_name("yushan777_SUPIR", create_if_missing=True), 
        "/root/clip-vit-large-patch14": modal.Volume.from_name("clip-vit-large-patch14", create_if_missing=True),
    },
)

class Model:
    @modal.enter()  #Enter the container
    def start_runtime(self):
        from models_init import initialize_models
        global model
        model = initialize_models()
        self.model = model
        print("Models initialized successfully")

    @modal.method()
    def generate_images_sync(self, request: UpscaleRequest) -> UpscaleResponse:
        request_id = str(int(time.time()))
        start_time = time.time()
        logger.info(f"Request {request_id} received at {start_time}")

        # Extract the target image from the nested input
        # Process image
        LQ_ips = base64_to_image(request.input["target_image"])
        LQ_img, h0, w0 = PIL2Tensor(LQ_ips, upsacle = request.upscaling_factor, min_size=args["min_size"])
        LQ_img = LQ_img.unsqueeze(0).to(SUPIR_device)[:, :3, :, :]
        captions = ['']

        try:
            # Diffusion Process
            samples = self.model.batchify_sample(
                LQ_img, captions,
                num_steps=args["edm_steps"],
                restoration_scale=args["s_stage1"],
                s_churn=args["s_churn"],
                s_noise=args["s_noise"],
                cfg_scale=args["s_cfg"],
                control_scale=args["s_stage2"],
                seed=args["seed"],
                num_samples=args["num_samples"],
                p_p=args["a_prompt"],
                n_p=args["n_prompt"],
                color_fix_type=args["color_fix_type"],
                use_linear_CFG=args["linear_CFG"],
                use_linear_control_scale=args["linear_s_stage2"],
                cfg_scale_start=args["spt_linear_CFG"],
                control_scale_start=args["spt_linear_s_stage2"]
            )
            for sample in samples:
                x = Tensor2PIL(sample, h0, w0)

            # Convert processed image to base64
            converted_image = image_to_base64(x) 
        except Exception as e:
            print(f"Error during inference: {e}")
            raise HTTPException(status_code=500, detail=f"Error during inference: {str(e)}")

        elapsed_time = time.time() - start_time
        logger.info(f"Request {request_id} completed in {elapsed_time:.2f} seconds")

        return UpscaleResponse(output={"image": f"data:image/png;base64,{converted_image}"})

# Instantiate the Modal Model class
app_model = Model()

# Define the endpoint
@web_app.post("/upscale_image", response_model=UpscaleResponse)
async def generate_images_endpoint(request: Request):
    try:
        # Parse the incoming JSON request into an ImageRequest object
        request_data = await request.json()
        image_request = UpscaleRequest(**request_data)

        # Call the generate_images method asynchronously
        image_response = await app_model.generate_images_sync.remote.aio(image_request)
        return image_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve the FastAPI app using Modal
@app.function(
    image=upscaler_image,
    gpu="t4",
    concurrency_limit=5,
    mounts=mounts,
    container_idle_timeout=120,  # in seconds
    volumes={
        "/root/laion_CLIP-ViT-bigG-14-laion2B-39B-b160k": modal.Volume.from_name("laion_CLIP-ViT-bigG-14-laion2B-39B-b160k", create_if_missing=True),  
        "/root/yushan777_SUPIR": modal.Volume.from_name("yushan777_SUPIR", create_if_missing=True), 
        "/root/clip-vit-large-patch14": modal.Volume.from_name("clip-vit-large-patch14", create_if_missing=True),
    },
)

@modal.asgi_app()
def fastapi_app():
    return web_app