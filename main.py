import os
import contextlib

import fastapi
import uvicorn
import pydantic

# The new streamlined Gemma API
from gemma import gm

app = fastapi.FastAPI(title="Gemma JAX Inference Service")

# Update to your downloaded Kaggle checkpoint path
CKPT_PATH = os.environ.get("CKPT_PATH", "/Users/hwasung_lee/Downloads/gemma-3-270m")

class ServerState:
    """Holds global state for the loaded model and sampler."""
    def __init__(self):
        self.sampler: gm.text.Sampler | None = None

state = ServerState()

class GenerateRequest(pydantic.BaseModel):
    prompt: str
    max_tokens: int = 128

class GenerateResponse(pydantic.BaseModel):
    text: str

@contextlib.asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    print("Initializing Model Architecture...")
    # NOTE: You must instantiate the architecture that matches your Kaggle download!
    # If you downloaded Gemma 3 4B:
    model = gm.nn.Gemma3_270M()
    # If you downloaded a Gemma 2 2B checkpoint, change this to: model = gm.nn.Gemma2_2B()

    print(f"Loading checkpoint from {CKPT_PATH}...")
    params = gm.ckpts.load_params(CKPT_PATH)

    print("Loading Tokenizer...")
    # For Gemma 3, the tokenizer is handled natively:
    tokenizer = gm.text.Gemma3Tokenizer()
    # (If using Gemma 2, load it manually: tokenizer = gm.text.Tokenizer("/path/to/tokenizer.model"))

    print("Initializing Sampler...")
    # The new Sampler API cleanly binds the model, weights, and tokenizer together
    state.sampler = gm.text.Sampler(
        model=model,
        params=params,
        tokenizer=tokenizer,
    )
    print("--- Model Loaded & Ready to Serve ---")

    yield

    print("Shutting down service...")

app.router.lifespan_context = lifespan

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    # The sampler now handles both string tokenization and the JIT compiled inference loop
    sampled_str = state.sampler.sample(
        request.prompt,
        max_new_tokens=request.max_tokens,
    )

    return GenerateResponse(text=sampled_str)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
