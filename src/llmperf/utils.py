import json
import math
import os
import random
import subprocess
import time
import torch
from typing import Any, Dict, Tuple
from pathlib import Path
from tensorrt_llm._utils import supports_inflight_batching  # noqa
from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.builder import get_engine_version
from transformers import LlamaTokenizerFast, AutoTokenizer, LlamaTokenizer, T5Tokenizer
from typing import List, Optional


RESULTS_VERSION = "2023-08-31"


class LLMPerfResults:
    def __init__(
        self,
        name: str,
        metadata: Dict[str, Any] = None,
    ):
        self.name = name
        self.metadata = metadata or {}
        self.timestamp = int(time.time())
        self.metadata["timestamp"] = self.timestamp
        self.version = RESULTS_VERSION

    def to_dict(self):
        data = {
            "version": self.version,
            "name": self.name,
        }
        data.update(self.metadata)
        data = flatten_dict(data)
        return data

    def json(self):
        data = self.to_dict()
        return json.dumps(data)


def upload_to_s3(results_path: str, s3_path: str) -> None:
    """Upload the results to s3.

    Args:
        results_path: The path to the results file.
        s3_path: The s3 path to upload the results to.

    """

    command = ["aws", "s3", "sync", results_path, f"{s3_path}/"]
    result = subprocess.run(command)
    if result.returncode == 0:
        print("Files uploaded successfully!")
    else:
        print("An error occurred:")
        print(result.stderr)


def randomly_sample_sonnet_lines_prompt(
    prompt_tokens_mean: int = 550,
    prompt_tokens_stddev: int = 250,
    expect_output_tokens: int = 150,
) -> Tuple[str, int]:
    """Generate a prompt that randomly samples lines from a the shakespeare sonnet at sonnet.txt.

    Args:
        prompt_length_mean: The mean length of the prompt to generate.
        prompt_len_stddev: The standard deviation of the length of the prompt to generate.
        expect_output_tokens: The number of tokens to expect in the output. This is used to
        determine the length of the prompt. The prompt will be generated such that the output
        will be approximately this many tokens.

    Note:
        tokens will be counted from the sonnet using the Llama tokenizer. Using one tokenizer
        ensures a fairer comparison across different LLMs. For example, if gpt 3.5 tokenizes
        a prompt in less tokens than Llama2, then this will be reflected in the results since
        they will be fed identical prompts.

    Returns:
        A tuple of the prompt and the length of the prompt.
    """

    tokenizer = LlamaTokenizerFast.from_pretrained(
        "hf-internal-testing/llama-tokenizer"
    )

    get_token_length = lambda text: len(tokenizer.encode(text))

    prompt = (
        "Randomly stream lines from the following text "
        f"with {expect_output_tokens} output tokens. "
        "Don't generate eos tokens:\n\n"
    )
    # get a prompt length that is at least as long as the base
    num_prompt_tokens = sample_random_positive_int(
        prompt_tokens_mean, prompt_tokens_stddev
    )
    while num_prompt_tokens < get_token_length(prompt):
        num_prompt_tokens = sample_random_positive_int(
            prompt_tokens_mean, prompt_tokens_stddev
        )
    remaining_prompt_tokens = num_prompt_tokens - get_token_length(prompt)
    sonnet_path = Path(__file__).parent.resolve() / "sonnet.txt"
    with open(sonnet_path, "r") as f:
        sonnet_lines = f.readlines()
    random.shuffle(sonnet_lines)
    sampling_lines = True
    while sampling_lines:
        for line in sonnet_lines:
            line_to_add = line
            if remaining_prompt_tokens - get_token_length(line_to_add) < 0:
                # This will cut off a line in the middle of a word, but that's ok since an
                # llm should be able to handle that.
                line_to_add = line_to_add[: int(math.ceil(remaining_prompt_tokens))]
                sampling_lines = False
                prompt += line_to_add
                break
            prompt += line_to_add
            remaining_prompt_tokens -= get_token_length(line_to_add)
    return (prompt, num_prompt_tokens)


def sample_random_positive_int(mean: int, stddev: int) -> int:
    """Sample random numbers from a gaussian distribution until a positive number is sampled.

    Args:
        mean: The mean of the gaussian distribution to sample from.
        stddev: The standard deviation of the gaussian distribution to sample from.

    Returns:
        A random positive integer sampled from the gaussian distribution.
    """
    ret = -1
    while ret <= 0:
        ret = int(random.gauss(mean, stddev))
    return ret


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class TensorRT:

    INTERNLM_META_INSTRUCTION = """You are an AI assistant whose name is InternLM (书生·浦语).
    - InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
    - InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
    """

    QWEN_PROMPT_TEMPLATE = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n"

    DEFAULT_PROMPT_TEMPLATES = {
        "InternLMForCausalLM": "<|User|>:{input_text}<eoh>\n<|Bot|>:",
        "InternLM2ForCausalLM": "<|im_start|>system\n"
        + INTERNLM_META_INSTRUCTION
        + "<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n",
        "QWenLMHeadModel": QWEN_PROMPT_TEMPLATE,
        "QWenForCausalLM": QWEN_PROMPT_TEMPLATE,
        "Qwen2ForCausalLM": QWEN_PROMPT_TEMPLATE,
        "Qwen2MoeForCausalLM": QWEN_PROMPT_TEMPLATE,
    }

    @staticmethod
    def read_decoder_start_token_id(engine_dir):
        with open(Path(engine_dir) / "config.json", "r") as f:
            config = json.load(f)
        return config["pretrained_config"]["decoder_start_token_id"]

    @staticmethod
    def read_model_name(engine_dir: str):
        engine_version = get_engine_version(engine_dir)

        with open(Path(engine_dir) / "config.json", "r") as f:
            config = json.load(f)

        if engine_version is None:
            return config["builder_config"]["name"], None

        model_arch = config["pretrained_config"]["architecture"]
        model_version = None
        if "GLM" in model_arch:
            model_version = config["pretrained_config"]["chatglm_version"]
        if "qwen" in model_arch.lower():
            model_version = config["pretrained_config"]["qwen_type"]
        return model_arch, model_version

    @staticmethod
    def throttle_generator(generator, stream_interval):
        for i, out in enumerate(generator):
            if not i % stream_interval:
                yield out

        if i % stream_interval:
            yield out

    @staticmethod
    def load_tokenizer(tokenizer_name_or_dir: str):
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=tokenizer_name_or_dir,
            token=os.environ.get("HF_ACCESS_TOKEN"),
        )
        tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})
        tokenizer.pad_token_id = 128002
        pad_id = tokenizer.pad_token_id
        end_id = tokenizer.eos_token_id
        return tokenizer, pad_id, end_id

    def prepare_enc_dec_inputs(
        self, batch_input_ids: List[torch.Tensor], engine_dir: str
    ):
        encoder_input_features = None

        encoder_input_ids = batch_input_ids
        decoder_start_token_id = self.read_decoder_start_token_id(
            os.path.join(engine_dir, "decoder")
        )
        decoder_input_ids = [
            torch.tensor([decoder_start_token_id], dtype=torch.int32)
            for _ in batch_input_ids
        ]
        encoder_output_lengths = None
        return (
            encoder_input_ids,
            encoder_input_features,
            encoder_output_lengths,
            decoder_input_ids,
        )
