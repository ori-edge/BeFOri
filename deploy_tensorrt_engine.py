import asyncio
import logging
import tensorrt_llm
import torch


from fastapi import FastAPI
from llmperf.utils import TensorRT
from queue import Empty
from ray import serve
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner

logger = logging.getLogger("ray.serve")

fastapi_app = FastAPI()


@serve.deployment
@serve.ingress(fastapi_app)
class DeployTensorRTEngine:
    def __init__(self, model_id: str, engine_dir: str, max_length: int):
        self.loop = asyncio.get_running_loop()

        self.model_id = model_id
        self.tokenizer, self.pad_id, self.end_id = TensorRT.load_tokenizer(
            tokenizer_name_or_dir=model_id
        )
        runner_cls = ModelRunner
        runtime_rank = tensorrt_llm.mpi_rank()
        runner_kwargs = dict(
            engine_dir=engine_dir,
            lora_dir=None,
            rank=runtime_rank,
            debug_mode=False,
            lora_ckpt_source="hf",
            gpu_weights_percent=1,
            max_output_len=max_length,
        )
        self.runner = runner_cls.from_dir(**runner_kwargs)
        self.runtime_rank = tensorrt_llm.mpi_rank()
        self.output_ids = []

    @fastapi_app.post("/")
    def handle_request(self, prompt: str, max_length: int):
        logger.info(f'Got prompt: "{prompt}"')
        self.loop.run_in_executor(None, self.generate_text, prompt, max_length)
        return self.output_ids

    def generate_text(self, prompt: str, max_length: int):
        input_ids = self.tokenizer([prompt], return_tensors="pt").input_ids
        with torch.no_grad():
            self.output_ids = self.runner.generate(
                batch_input_ids=input_ids,
                encoder_input_ids=None,
                encoder_input_features=None,
                encoder_output_lengths=None,
                max_new_tokens=max_length,
                max_attention_window_size=None,
                sink_token_length=None,
                end_id=self.end_id,
                pad_id=self.pad_id,
                temperature=1.0,
                top_k=1,
                top_p=0.0,
                num_beams=1,
                num_return_sequences=None,
                length_penalty=1.0,
                early_stopping=1,
                repetition_penalty=1.0,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                stop_words_list=None,
                bad_words_list=None,
                output_cum_log_probs=False,
                output_log_probs=False,
                random_seed=0,
                lora_uids=None,
                prompt_table=None,
                streaming=False,
                output_sequence_lengths=True,
                no_repeat_ngram_size=None,
                return_dict=True,
                medusa_choices=None,
                return_all_generated_tokens=True,
                input_token_extra_ids=None,
            )
            torch.cuda.synchronize()

    async def consume_streamer(self, streaming_interval):
        while True:
            try:
                for curr_outputs in self.output_ids:
                    print(
                        f"Consuming streamer, found current outputs: \n{curr_outputs}"
                    )
                    if self.runtime_rank == 0:
                        _output_ids = curr_outputs["output_ids"]
                        for _id in _output_ids:
                            yield _id
                break
            except Empty:
                # The streamer raises an Empty exception if the next token
                # hasn't been generated yet. `await` here to yield control
                # back to the event loop so other coroutines can run.
                await asyncio.sleep(0.001)


app = DeployTensorRTEngine.bind("meta-llama/Meta-Llama-3.1-8B-Instruct", "/home/ubuntu/BeFOri/tensorrt/output/trt_engines/", 152)
