import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

MAX_WORKERS = 1
QUEUE_SIZE = 3