import os
import time
from typing import Any, Dict, Tuple
import ray
import requests

from llmperf.ray_llm_client import LLMClient
from llmperf.models import RequestConfig
from llmperf import common_metrics
from llmperf.utils import construct_transformers_args

@ray.remote
class LocalVLLMClient(LLMClient):
    """Client for local models deployments via vllm for Chat Completions"""

    def llm_request(self, request_config: RequestConfig) -> Tuple[Dict[str, Any], str, RequestConfig]:
        # Prepare to collect metrics
        tokens_received = 0
        ttft = 0
        generated_text = ""
        metrics = {}

        metrics[common_metrics.ERROR_CODE] = None
        metrics[common_metrics.ERROR_MSG] = ""

        # Prepare to call the fastapi endpoint
        url = "http://localhost:8000/"
        prompt_arg = request_config.prompt
        prompt, prompt_len = prompt_arg
        generator_args = {"prompt": prompt,
                          "max_length": request_config.sampling_params["max_tokens"]}

        # Call the endpoint with the provided prompt and max length params
        try:
            start_time = time.monotonic()
            response = requests.post(url, params=generator_args, stream=True)
            response.raise_for_status()

            first_token = True
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                generated_text += chunk
                print(chunk, end="")
                if first_token is True:
                    ttft = time.monotonic() - start_time
                    first_token = False
        except Exception as e:
            metrics[common_metrics.ERROR_MSG] = str(e)
            print(f"Warning Or Error: {e}")
        total_request_time =  time.monotonic() - start_time
        #TODO
        breakpoint()
        tokens_received = 0
        output_throughput = tokens_received / total_request_time
        metrics[common_metrics.INTER_TOKEN_LAT] = (
                                                          total_request_time - ttft
                                                  ) / tokens_received
        metrics[common_metrics.TTFT] = ttft
        metrics[common_metrics.E2E_LAT] = total_request_time
        metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = output_throughput
        metrics[common_metrics.NUM_TOTAL_TOKENS] = tokens_received + prompt_len
        metrics[common_metrics.NUM_OUTPUT_TOKENS] = tokens_received
        metrics[common_metrics.NUM_INPUT_TOKENS] = prompt_len

        return metrics, generated_text, request_config
