from tensorrt_llm import LLM, SamplingParams
import logging
from fastapi import FastAPI, HTTPException
from ray import serve
from typing import List, Dict
import uuid
import time
import threading

logger = logging.getLogger("ray.serve")

fastapi_app = FastAPI()


@serve.deployment(ray_actor_options={"num_gpus": 1})
@serve.ingress(fastapi_app)
class DeployTRTEngine:
    def __init__(self, model_id: str):
        self.model = LLM(model=model_id)
        self.queue = {}
        self.statuses = {}
        self.outputs = []
        self.timer = 0

    @fastapi_app.post("/")
    def handle_request(self, prompt: str, ccr: int):
        # If the queue is empty then (re)start the timer
        queue_len = len(self.queue)
        if queue_len == 0:
            self.timer = time.time()

        # Generate a unique ID for the request and set up tracking
        task_id = str(uuid.uuid4())
        self.queue[task_id] = prompt
        queue_len += 1
        self.statuses[task_id] = "in queue"

        # If we have the desired number of concurrent requests or 2 seconds have passed then start generating
        if queue_len >= ccr or time.time() - self.timer > 2:
            prompts = []
            # make a dictionary of prompts that contain the desired number of concurrent requests or less
            while len(prompts) < min(ccr, queue_len):
                _task_id = next(iter(self.queue))
                _prompt = self.queue.pop(_task_id)
                prompts.append({_task_id: prompt})
                self.statuses[_task_id] = "in progress"

            # Start a background thread to process the task
            threading.Thread(target=self.generate_text, args=prompts).start()
        return {"task_id": task_id}

    def generate_text(self, prompts: List[Dict[str, str]]):
        prompt_list = [list(d.values())[0] for d in prompts]
        raw_outputs = self.model.generate(prompt_list)

        for _output in raw_outputs:
            prompt_dict = prompts.pop(0)
            _task_id, input_prompt = next(iter(prompt_dict.items()))
            self.outputs[_task_id] = {
                "prompt": input_prompt,
                "text": _output.output[0].text,
                "token_len": len(_output.output[0].token_ids),
            }
            self.statuses[_task_id] = "complete"

    @fastapi_app.get("/response/{task_id}")
    def get_response(self, task_id: str):
        # Get the status of the task, if the task id is not found raise an error
        try:
            status = self.statuses.pop(task_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Task ID not found")

        if status in ["in queue", "in progress"]:
            raise HTTPException(status_code=202, detail=f"Task is {status}.")
        ret = self.outputs.pop(task_id)
        return ret


app = DeployTRTEngine.bind("meta-llama/Meta-Llama-3.1-8B-Instruct")
