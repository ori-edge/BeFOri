import argparse
import asyncio
import logging
import os
from queue import Empty

from fastapi import FastAPI
from starlette.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from torch import bfloat16
from ray import serve

logger = logging.getLogger("ray.serve").setLevel(logging.WARNING)

fastapi_app = FastAPI()


@serve.deployment
@serve.ingress(fastapi_app)
class DeployVllm:
    def __init__(self, model_id: str):
        self.loop = asyncio.get_running_loop()

        self.model_id = model_id
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        self.access_token = os.environ.get("HF_ACCESS_TOKEN")
        # TODO: update cli args to pass parameters to model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, token=self.access_token, torch_dtype=bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, token=self.access_token
        )

    @fastapi_app.post("/")
    def handle_request(self, prompt: str, max_length: int) -> StreamingResponse:
        logger.info(f'Got prompt: "{prompt}"')
        streamer = TextIteratorStreamer(
            self.tokenizer, timeout=0, skip_prompt=True, skip_special_tokens=True
        )
        self.loop.run_in_executor(
            None, self.generate_text, prompt, max_length, streamer
        )
        return StreamingResponse(
            self.consume_streamer(streamer), media_type="text/plain"
        )

    def generate_text(
        self, prompt: str, max_length: int, streamer: TextIteratorStreamer
    ):
        input_ids = self.tokenizer([prompt], return_tensors="pt").input_ids
        self.model.generate(input_ids, streamer=streamer, max_length=max_length)

    @staticmethod
    async def consume_streamer(streamer: TextIteratorStreamer):
        while True:
            try:
                for token in streamer:
                    logger.info(f'Yielding token: "{token}"')
                    yield token
                break
            except Empty:
                # The streamer raises an Empty exception if the next token
                # hasn't been generated yet. `await` here to yield control
                # back to the event loop so other coroutines can run.
                await asyncio.sleep(0.001)


app = DeployVllm.bind("meta-llama/Meta-Llama-3.1-8B-Instruct")
