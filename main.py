import contextlib

import fastapi
import uvicorn
import pydantic
import asyncio
import argparse
import logging
import json
import pathlib
import typing
import uuid

from gemma import gm


LOGGER = logging.getLogger(__name__)


class GenerateRequest(pydantic.BaseModel):
    request_id: str = pydantic.Field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str
    max_tokens: int = 128


class GenerateResponse(pydantic.BaseModel):
    request_id: str
    text: str


class QueueItem(typing.NamedTuple):
    prompt: str
    max_tokens: int
    future: asyncio.Future


async def dynamic_batch_worker(
    sampler: gm.text.Sampler,
    queue: asyncio.Queue,
    max_batch_size: int,
    batch_timeout_secs: float,
):
    """Continuously monitors the queue and processes batches."""
    while True:
        batch: list[QueueItem] = [await queue.get()]
        await asyncio.sleep(batch_timeout_secs)
        while not queue.empty() and len(batch) < max_batch_size:
            batch.append(queue.get_nowait())

        valid_batch = [item for item in batch if not item.future.cancelled()]
        if not valid_batch:
            continue

        # Extract the prompts and futures from our grouped batch
        prompts: list[str] = [item.prompt for item in valid_batch]
        max_tokens_s: list[int] = [item.max_tokens for item in valid_batch]
        futures: list[asyncio.Future] = [item.future for item in valid_batch]

        # We need to sample until we satisfy all the requests in the batch,
        # in this case we should sample until the absolute max.
        batch_max_tokens = max(max_tokens_s)

        try:
            # 4. Run the synchronous JAX math in a background thread
            # so it doesn't freeze the async queue!
            sampled_strs: list[str] = await asyncio.to_thread(
                sampler.sample, prompts, max_new_tokens=batch_max_tokens
            )  # type: ignore[invalid-assignment]

            # 5. Hand the results back to the waiting HTTP requests
            for fut, result_text, max_tokens in zip(
                futures, sampled_strs, max_tokens_s
            ):
                if not fut.done():
                    fut.set_result(result_text[:max_tokens])

        except Exception as e:
            # If JAX crashes, send the error to all waiting clients
            for fut in futures:
                if not fut.done():
                    fut.set_exception(e)


@contextlib.asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    checkpoint_path = app.state.args.checkpoint_path
    max_batch_size = app.state.args.max_batch_size
    batch_timeout_secs = app.state.args.batch_timeout_secs
    max_queue_size = app.state.args.max_queue_size

    LOGGER.info("Initialising model architecture...")
    model = gm.nn.Gemma3_270M()
    params = gm.ckpts.load_params(checkpoint_path)
    tokenizer = gm.text.Gemma3Tokenizer()
    sampler = gm.text.Sampler(
        model=model,  # type: ignore[invalid-argument-type]
        params=params,
        tokenizer=tokenizer,
    )
    LOGGER.info("Model has been loaded and ready to serve!")

    def _on_worker_done(task: asyncio.Task) -> None:
        if not task.cancelled():
            if exc := task.exception():
                LOGGER.critical("Batch worker died unexpectedly: %s", exc, exc_info=exc)

    request_queue = asyncio.Queue(maxsize=max_queue_size)
    worker_task = asyncio.create_task(
        dynamic_batch_worker(sampler, request_queue, max_batch_size, batch_timeout_secs)
    )
    worker_task.add_done_callback(_on_worker_done)

    yield {"queue": request_queue}

    LOGGER.info("Shutting down service...")
    worker_task.cancel()


app = fastapi.FastAPI(title="Gemma JAX Inference Service", lifespan=lifespan)


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: fastapi.Request, payload: GenerateRequest):
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    try:
        request.state.queue.put_nowait(
            QueueItem(payload.prompt, payload.max_tokens, future)
        )
    except asyncio.QueueFull:
        LOGGER.warning("Queue is full. Rejecting request with 429.")
        raise fastapi.HTTPException(
            status_code=429,
            detail="Server is currently overloaded. Please try again later.",
            headers={"Retry-After": "5"},  # Tells polite clients to wait 5 seconds
        )

    try:
        return GenerateResponse(
            text=await asyncio.wait_for(future, 300), request_id=payload.request_id
        )
    except asyncio.CancelledError:
        LOGGER.debug("Client cancelled request: %s", payload.request_id)
        future.cancel()
        raise  # If no raise is sent here FastAPI tries to send a None response to the client.
    except asyncio.TimeoutError:
        LOGGER.error("Timed out processing request: %s", payload.request_id)
        raise fastapi.HTTPException(
            status_code=fastapi.status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Inference engine timed out: {payload.request_id}",
        )
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
        default=str(pathlib.Path.home() / "Downloads" / "gemma-3-270m"),
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
