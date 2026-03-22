import os
import contextlib

import fastapi
import uvicorn
import pydantic
import asyncio

# The new streamlined Gemma API
from gemma import gm

# Update to your downloaded Kaggle checkpoint path
CKPT_PATH = os.environ.get("CKPT_PATH", "/Users/hwasung_lee/Downloads/gemma-3-270m")
BATCH_TIMEOUT_SECS = 0.05
MAX_BATCH_SIZE = 4


class GenerateRequest(pydantic.BaseModel):
    prompt: str
    max_tokens: int = 128


class GenerateResponse(pydantic.BaseModel):
    text: str


async def dynamic_batch_worker(sampler, queue: asyncio.Queue):
    """Continuously monitors the queue and processes batches."""
    while True:
        # 1. Wait until at least one request is in the queue
        prompt, future, max_tokens = await queue.get()
        batch = [(prompt, future, max_tokens)]

        # 2. Wait a tiny fraction of a second to let other requests pile up
        await asyncio.sleep(BATCH_TIMEOUT_SECS)

        # 3. Scoop up any other requests that arrived during the wait
        while not queue.empty() and len(batch) < MAX_BATCH_SIZE:
            batch.append(queue.get_nowait())

        # Extract the prompts and futures from our grouped batch
        prompts = [item[0] for item in batch]
        futures = [item[1] for item in batch]

        # Use the max_tokens from the first request in the batch (simplified)
        batch_max_tokens = batch[0][2]

        try:
            # 4. Run the synchronous JAX math in a background thread
            # so it doesn't freeze the async queue!
            sampled_strs = await asyncio.to_thread(
                sampler.sample, prompts, max_new_tokens=batch_max_tokens
            )

            # 5. Hand the results back to the waiting HTTP requests
            for fut, result_text in zip(futures, sampled_strs):
                if not fut.done():
                    fut.set_result(result_text)

        except Exception as e:
            # If JAX crashes, send the error to all waiting clients
            for fut in futures:
                if not fut.done():
                    fut.set_exception(e)


@contextlib.asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    print("Initializing Model Architecture...")
    model = gm.nn.Gemma3_270M()

    print(f"Loading checkpoint from {CKPT_PATH}...")
    params = gm.ckpts.load_params(CKPT_PATH)

    print("Loading Tokenizer...")
    tokenizer = gm.text.Gemma3Tokenizer()

    print("Initializing Sampler...")
    # The new Sampler API cleanly binds the model, weights, and tokenizer together
    sampler = gm.text.Sampler(
        model=model,  # type: ignore[invalid-argument-type]
        params=params,
        tokenizer=tokenizer,
    )
    print("--- Model Loaded & Ready to Serve ---")

    request_queue = asyncio.Queue()
    worker_task = asyncio.create_task(dynamic_batch_worker(sampler, request_queue))

    yield {"queue": request_queue}

    print("Shutting down service...")
    worker_task.cancel()


app = fastapi.FastAPI(title="Gemma JAX Inference Service", lifespan=lifespan)


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: fastapi.Request, payload: GenerateRequest):
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    await request.state.queue.put((payload.prompt, future, payload.max_tokens))

    try:
        return GenerateResponse(text=await future)
    except Exception as e:
        print(f"Generation Error: {str(e)}")
        raise fastapi.HTTPException(
            status_code=500, detail=f"Inference engine failed: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
