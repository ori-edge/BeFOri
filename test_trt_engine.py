import logging
import tensorrt_llm
import torch

from llmperf.utils import TensorRT
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner

logger = logging.getLogger("ray.serve")


class DeployTensorRTEngine:
    def __init__(self, model_id: str, engine_dir: str, max_length: int):
        self.model_id = model_id
        self.tokenizer, self.pad_id, self.end_id = TensorRT.load_tokenizer(
            tokenizer_name_or_dir=model_id
        )
        self.tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})
        self.tokenizer.pad_token_id = 128002
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

    def handle_request(self, prompt: str, max_length: int):
        logger.info(f'Got prompt: "{prompt}"')
        input_ids = self.tokenizer([prompt], return_tensors="pt").input_ids
        with torch.no_grad():
            outputs = self.runner.generate(
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
                streaming=True,
                output_sequence_lengths=True,
                no_repeat_ngram_size=None,
                return_dict=True,
                medusa_choices=None,
                return_all_generated_tokens=True,
                input_token_extra_ids=None,
            )
            torch.cuda.synchronize()
        for curr_outputs in self.throttle_generator(outputs, 1):
            if self.runtime_rank == 0:
                output_ids = curr_outputs['output_ids'][0][0]
                output_text = self.tokenizer.decode(output_ids)
                breakpoint()
        return output_text

    @staticmethod
    def throttle_generator(generator, stream_interval):
        for i, out in enumerate(generator):
            if not i % stream_interval:
                yield out

        if i % stream_interval:
            yield out

if __name__ == "__main__":
    max_length = 152
    prompt = "Why is this so hard?"
    TRT = DeployTensorRTEngine(model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
                               engine_dir="/home/ubuntu/BeFOri/tensorrt/output/trt_engines/",
                               max_length=max_length)
    output_text = TRT.handle_request(prompt=prompt, max_length=max_length)
    breakpoint()
    print(output_text)
