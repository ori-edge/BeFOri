from tensorrt_llm import LLM, SamplingParams
import logging
from fastapi import FastAPI
from ray import serve

logger = logging.getLogger("ray.serve")

fastapi_app = FastAPI()

@serve.deployment(ray_actor_options={"num_gpus": 1})
@serve.ingress(fastapi_app)
class DeployTRTEngine:
    def __init__(self, model_id: str):
        self.model = LLM(model=model_id)

    @fastapi_app.post("/")
    def handle_request(self, prompt: str):

        prompts = [prompt]

        outputs = self.model.generate(prompts)
        # if outputs[0].request_id == request_id
        prompt = outputs[0].prompt
        generated_text = outputs[0].outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        return prompt, generated_text


app = DeployTRTEngine.bind("meta-llama/Meta-Llama-3.1-8B-Instruct")
