import os
import contextlib

import fastapi
import uvicorn
import pydantic
import sentencepiece as spm

from gemma import params as params_lib
from gemma import sampler
from gemma import transformer

app = fastapi.FastAPI(title="Gemma JAX Inference Service")

# --- Configuration ---
# Update these paths to your downloaded Kaggle checkpoint and tokenizer
CKPT_PATH = os.environ.get("CKPT_PATH", "/path/to/kaggle/checkpoint/2b-it")
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "/path/to/kaggle/tokenizer.model")
CACHE_SIZE = int(os.environ.get("CACHE_SIZE", "1024"))

class ServerState:
    """Holds global state for the loaded model and sampler."""
    def __init__(self):
        self.vocab = None
        self.gemma_sampler = None

state = ServerState()

class GenerateRequest(pydantic.BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.0  # Greedy decoding by default

class GenerateResponse(pydantic.BaseModel):
    text: str

@contextlib.asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    """Lifecycle manager to load model weights into device memory on startup."""
    print("Loading SentencePiece tokenizer...")
    state.vocab = spm.SentencePieceProcessor()
    state.vocab.Load(TOKENIZER_PATH)

    print("Loading and formatting Flax parameters from checkpoint...")
    # This automatically handles the Orbax/Flax parameters
    params = params_lib.load_and_format_params(CKPT_PATH)

    print("Configuring Transformer...")
    # Infer architecture config from the checkpoint parameters.
    # The cache_size defines the static maximum sequence length (prompt + completion)
    # to avoid JAX recompilation.
    model_config = transformer.TransformerConfig.from_params(
        params,
        cache_size=CACHE_SIZE
    )

    print("Initializing JAX Sampler...")
    # The Sampler handles the JIT-compiled inference loop internally
    state.gemma_sampler = sampler.Sampler(
        transformer=transformer.Transformer(model_config),
        vocab=state.vocab,
        params=params['transformer'],
    )
    print("--- Model Loaded & Ready to Serve ---")

    yield

    print("Shutting down service...")

app.router.lifespan_context = lifespan

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    # The sampler expects a list of prompt strings
    prompts = [request.prompt]

    # Note: The very first request will take significantly longer because
    # XLA has to JIT compile the forward pass.
    sampled_strs = state.gemma_sampler(
        input_strings=prompts,
        total_generation_steps=request.max_tokens,
    )

    return GenerateResponse(text=sampled_strs[0])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
