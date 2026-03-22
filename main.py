import os
import contextlib

import fastapi
import uvicorn
import pydantic
import asyncio
import argparse
import logging
import json

from gemma import gm


LOGGER = logging.getLogger(__name__)


class GenerateRequest(pydantic.BaseModel):
    prompt: str
    max_tokens: int = 128


class GenerateResponse(pydantic.BaseModel):
    text: str


async def dynamic_batch_worker(
    sampler, queue: asyncio.Queue, max_batch_size: int, batch_timeout_secs: float
):
    """Continuously monitors the queue and processes batches."""
    while True:
        # 1. Wait until at least one request is in the queue
        prompt, future, max_tokens = await queue.get()
        batch = [(prompt, future, max_tokens)]

        # 2. Wait a tiny fraction of a second to let other requests pile up
        await asyncio.sleep(batch_timeout_secs)

        # 3. Scoop up any other requests that arrived during the wait
        while not queue.empty() and len(batch) < max_batch_size:
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
    checkpoint_path = getattr(app.state.args, "checkpoint_path")
    max_batch_size = getattr(app.state.args, "max_batch_size")
    batch_timeout_secs = getattr(app.state.args, "batch_timeout_secs")
    max_queue_size = getattr(app.state.args, "max_queue_size")

    LOGGER.info("Initialising model architecture...")
    model = gm.nn.Gemma3_270M()
    params = gm.ckpts.load_params(checkpoint_path)
    tokenizer = gm.text.Gemma3Tokenizer()
    sampler = gm.text.Sampler(
        model=model,  # type: ignore[invalid-argument-type]
        params=params,
        tokenizer=tokenizer,
    )
    LOGGER.info("Model hav been loaded and ready to serve!")

    request_queue = asyncio.Queue(maxsize=max_queue_size)
    worker_task = asyncio.create_task(
        dynamic_batch_worker(sampler, request_queue, max_batch_size, batch_timeout_secs)
    )

    yield {"queue": request_queue}

    LOGGER.info("Shutting down service...")
    worker_task.cancel()


app = fastapi.FastAPI(title="Gemma JAX Inference Service", lifespan=lifespan)


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: fastapi.Request, payload: GenerateRequest):
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    try:
       request.state.queue.put_nowait((payload.prompt, future, payload.max_tokens))
    except asyncio.QueueFull:
        LOGGER.warning("Queue is full. Rejecting requst with 429.")
        raise fastapi.HTTPException(
            status_code=429,
            detail="Server is currently overloaded. Please try again later.",
            headers={"Retry-After": "5"} # Tells polite clients to wait 5 seconds
        )

    try:
        return GenerateResponse(text=await future)
    except Exception as e:
        LOGGER.exception("Generation error.")
        raise fastapi.HTTPException(
            status_code=500, detail=f"Inference engine failed: {str(e)}"
        )


def main(args: argparse.Namespace) -> None:
    """Main function."""
    LOGGER.info("Running with args: %s", json.dumps(dict(vars(args)), indent=2))
    app.state.args = args
    uvicorn.run(app, host="0.0.0.0", port=8000)


def arg_parse() -> argparse.Namespace:
    """Parse arguments and return an args object."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-level",
        dest="log_level",
        choices=logging._nameToLevel.keys(),
        default="INFO",
        help="Log level for the default logger.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="/Users/hwasung_lee/Downloads/gemma-3-270m",
        help="Path to the downloaded Gemma checkpoint directory",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=4,
        help="Maximum number of requests to process in a single JAX batch",
    )
    parser.add_argument(
        "--batch-timeout-secs",
        type=float,
        default=0.05,
        help="Time in seconds to wait for additional requests to batch",
    )
    parser.add_argument(
        "--max-queue-size",
        type=int,
        default=50,
        help="Maximum number of pending requests before rejecting with 429",
    )
    return parser.parse_args()


if __name__ == "__main__":
    ARGS = arg_parse()
    logging.basicConfig(
        level=logging._nameToLevel[ARGS.log_level],
        format=("%(asctime)s " + logging.BASIC_FORMAT),
    )
    main(ARGS)
