import os
import contextlib

import fastapi
import uvicorn
import pydantic

# The new streamlined Gemma API
from gemma import gm

# Update to your downloaded Kaggle checkpoint path
CKPT_PATH = os.environ.get("CKPT_PATH", "/Users/hwasung_lee/Downloads/gemma-3-270m")

class GenerateRequest(pydantic.BaseModel):
    prompt: str
    max_tokens: int = 128

class GenerateResponse(pydantic.BaseModel):
    text: str

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
        model=model,
        params=params,
        tokenizer=tokenizer,
    )
    print("--- Model Loaded & Ready to Serve ---")

    yield {"sampler": sampler}

    print("Shutting down service...")

app = fastapi.FastAPI(title="Gemma JAX Inference Service", lifespan=lifespan)

@app.post("/generate", response_model=GenerateResponse)
def generate(request: fastapi.Request, payload: GenerateRequest):
    # The sampler now handles both string tokenization and the JIT compiled inference loop
    sampled_str = request.state.sampler.sample(
        payload.prompt,
        max_new_tokens=payload.max_tokens,
    )

    return GenerateResponse(text=sampled_str)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
