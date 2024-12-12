from SUPIR.util import create_SUPIR_model,convert_dtype
from config import args
import torch.cuda

if torch.cuda.is_available():
    SUPIR_device = 'cuda:0'
else:
    raise ValueError('Currently support CUDA only.')

def initialize_models():
    # Load SUPIR
    model = create_SUPIR_model('SUPIR_v0.yaml', SUPIR_sign=args["SUPIR_sign"])
    if args["loading_half_params"]:
        model = model.half()
    if args["use_tile_vae"]:
        model.init_tile_vae(encoder_tile_size=args["encoder_tile_size"], decoder_tile_size=args["decoder_tile_size"])
    model.ae_dtype = convert_dtype(args["ae_dtype"])
    model.model.dtype = convert_dtype(args["diff_dtype"])
    model = model.to(SUPIR_device)
    return model

