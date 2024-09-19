import asyncio
import logging
import os
import time
from queue import Empty
from typing import Dict, Any, Tuple

from fastapi import FastAPI
from starlette.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, LlamaTokenizerFast

from ray import serve

from llmperf.ray_llm_client import LLMClient
from llmperf.models import RequestConfig
from llmperf.utils import construct_transformers_args
from llmperf import common_metrics


logger = logging.getLogger("ray.serve")

fastapi_app = FastAPI()


@serve.deployment
@serve.ingress(fastapi_app)
class TransformersLibClient(LLMClient):
    def __init__(self,  request_config: RequestConfig) -> Dict[str, Any]:
        self.loop = asyncio.get_running_loop()
        self.access_token = os.environ.get("HF_ACCESS_TOKEN")
        self.model = None
        self.tokenizer = None
        self.llama_tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")

    # @fastapi_app.post("/")
    def llm_request(self, request_config: RequestConfig) -> Tuple[Dict[str, Any], str, RequestConfig]:
        if self.model is None:
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
            model_args = construct_transformers_args(request_config=request_config)
            model_args["token"] = self.access_token

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                request_config.model, token=self.access_token
            )

        max_length = request_config.sampling_params["max_tokens"]
        prompt_tup = request_config.prompt
        prompt, prompt_len = prompt_tup
        logger.info(f'Got prompt: "{prompt}"')
        tokens_received = 0
        ttft = 0
        metrics = {common_metrics.ERROR_CODE: None, common_metrics.ERROR_MSG: ""}

        try:
            streamer = TextIteratorStreamer(
                self.tokenizer, timeout=0, skip_prompt=True, skip_special_tokens=True
            )
            start_time = time.monotonic()
            self.loop.run_in_executor(None, self.generate_text, prompt, streamer, max_length)
            #TODO: split this section into a separate function - add fastapi back?
            # streamer = StreamingResponse(
            #     self.consume_streamer(streamer), media_type="text/plain"
            # )
            generated_text = ""
            first_token = True
            for new_text in streamer:
                generated_text += new_text
                if first_token is True:
                    ttft = time.monotonic() - start_time
                    first_token = False
            total_request_time = time.monotonic() - start_time

        except Exception as e:
            metrics[common_metrics.ERROR_MSG] = str(e)
            print(f"Warning Or Error: {e}")

        return metrics, generated_text, request_config

    def generate_text(self, prompt: str, streamer: TextIteratorStreamer, max_tokens: int):
        input_ids = self.tokenizer([prompt], return_tensors="pt").input_ids
        self.model.generate(input_ids, streamer=streamer, max_length=max_tokens)

    async def consume_streamer(self, streamer: TextIteratorStreamer):
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
