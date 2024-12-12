from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models_init import initialize_models,SUPIR_device
from models import UpscaleRequest, UpscaleResponse
from fastapi import HTTPException
from config import logger, MAX_WORKERS, QUEUE_SIZE
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from utils import base64_to_image, image_to_base64
from SUPIR.util import PIL2Tensor,Tensor2PIL
from config import args

app = FastAPI(
    title="Upscaler API",
    description="API for Upscaling images using SUPIR",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = initialize_models()
logger.info("Models initialized successfully")

executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)  # Single worker to ensure sequential processing
request_queue = asyncio.Queue(maxsize=QUEUE_SIZE)  # Queue to hold incoming requests with a maximum size of 2

# Background worker to process the queue sequentially
async def process_queue():
    while True:
        request_id, request, future = await request_queue.get()
        try:
            # Process the request (offloaded to a separate thread if blocking)
            result = await asyncio.get_event_loop().run_in_executor(executor, generate_images_sync, request)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
        finally:
            request_queue.task_done()

# Start the background worker at startup
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_queue())

# Function to handle image generation
def generate_images_sync(request: UpscaleRequest) -> UpscaleResponse:
    request_id = str(int(time.time()))
    start_time = time.time()
    logger.info(f"Request {request_id} received at {start_time}")

    # Extract the target image from the nested input
    LQ_ips = base64_to_image(request.input["target_image"])
    LQ_img, h0, w0 = PIL2Tensor(LQ_ips, upsacle = request.upscaling_factor, min_size=args["min_size"])
    LQ_img = LQ_img.unsqueeze(0).to(SUPIR_device)[:, :3, :, :]
    captions = ['']

    try:
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
            x = Tensor2PIL(sample, h0, w0)
        # Convert processed image to base64
        converted_image = image_to_base64(x) 
    except Exception as e:
        print(f"Error during inference: {e}")
        raise HTTPException(status_code=500, detail=f"Error during inference: {str(e)}")


    elapsed_time = time.time() - start_time
    logger.info(f"Request {request_id} completed in {elapsed_time:.2f} seconds")

    return UpscaleResponse(output={"image": f"data:image/png;base64,{converted_image}"})


        # Endpoint for image generation
@app.post("/upscale_image/", response_model=UpscaleResponse)
async def generate_images(request: UpscaleRequest):
    request_id = str(int(time.time()))
    future = asyncio.get_event_loop().create_future()  # Create a future to hold the result

    try:
        # Attempt to add request to queue with a timeout
        await asyncio.wait_for(request_queue.put((request_id, request, future)), timeout=5.0)
    except asyncio.TimeoutError:
        # Return error if the queue is full and timeout is reached
        raise HTTPException(status_code=503, detail="Service is currently busy. Please try again later.")
    
    return await future  # Await the result from the queue processor

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "endpoint:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # Since using GPU, multiple workers might not be beneficial
        log_level="info"
    )  