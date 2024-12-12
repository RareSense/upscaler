import torch.cuda
from SUPIR.util import create_SUPIR_model, PIL2Tensor, Tensor2PIL, convert_dtype
from PIL import Image
import os

if torch.cuda.is_available():
    SUPIR_device = 'cuda:0'
else:
    raise ValueError('Currently support CUDA only.')

# Hardcoded parameters
img_path = "123.jpg"  # Replace with the path to your image
upscale = 2  # Replace with your desired upscale factor

# Other parameters
args = {
    "SUPIR_sign": "Q",
    "seed": 1234,
    "min_size": 1024,
    "edm_steps": 50,
    "s_stage1": -1,
    "s_churn": 5,
    "s_noise": 1.003,
    "s_cfg": 7.5,
    "s_stage2": 1.0,
    "num_samples": 1,
    "a_prompt": "",
    "n_prompt": "",
    "color_fix_type": "Wavelet",
    "linear_CFG": True,
    "linear_s_stage2": False,
    "spt_linear_CFG": 4.0,
    "spt_linear_s_stage2": 0.0,
    "ae_dtype": "bf16",
    "diff_dtype": "fp16",
    "no_llava": False,
    "loading_half_params": False,
    "use_tile_vae": False,
    "encoder_tile_size": 512,
    "decoder_tile_size": 64,
    "load_8bit_llava": False,
}

# Load SUPIR
model = create_SUPIR_model('SUPIR_v0.yaml', SUPIR_sign=args["SUPIR_sign"])
if args["loading_half_params"]:
    model = model.half()
if args["use_tile_vae"]:
    model.init_tile_vae(encoder_tile_size=args["encoder_tile_size"], decoder_tile_size=args["decoder_tile_size"])
model.ae_dtype = convert_dtype(args["ae_dtype"])
model.model.dtype = convert_dtype(args["diff_dtype"])
model = model.to(SUPIR_device)

# Process single image
img_name = os.path.splitext(os.path.basename(img_path))[0]
LQ_ips = Image.open(img_path)
LQ_img, h0, w0 = PIL2Tensor(LQ_ips, upsacle=upscale, min_size=args["min_size"])
LQ_img = LQ_img.unsqueeze(0).to(SUPIR_device)[:, :3, :, :]
captions = ['']

# Step 3: Diffusion Process
samples = model.batchify_sample(
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
    Tensor2PIL(sample, h0, w0).save('generate.jpg')