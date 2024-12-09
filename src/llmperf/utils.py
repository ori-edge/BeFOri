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
    DEFAULT_HF_MODEL_DIRS = {
        'BaichuanForCausalLM': 'baichuan-inc/Baichuan-13B-Chat',
        'BaiChuanForCausalLM': 'baichuan-inc/Baichuan-13B-Chat',
        'BloomForCausalLM': 'bigscience/bloom-560m',
        'GLMModel': 'THUDM/glm-10b',
        'ChatGLMModel': 'THUDM/chatglm3-6b',
        'ChatGLMForCausalLM': 'THUDM/chatglm3-6b',
        'RWForCausalLM': 'tiiuae/falcon-rw-1b',
        'FalconForCausalLM': 'tiiuae/falcon-rw-1b',
        'GPT2LMHeadModel': 'gpt2',
        'GPT2LMHeadCustomModel': 'gpt2',
        'Starcoder2ForCausalLM': 'bigcode/starcoder2-3b',
        'GPTForCausalLM': 'gpt2',
        'GPTJForCausalLM': 'EleutherAI/gpt-j-6b',
        'GPTNeoXForCausalLM': 'EleutherAI/gpt-neox-20b',
        'InternLMForCausalLM': 'internlm/internlm-chat-7b',
        'InternLM2ForCausalLM': 'internlm/internlm2-chat-7b',
        'LlamaForCausalLM': 'meta-llama/Llama-2-7b-hf',
        'MPTForCausalLM': 'mosaicml/mpt-7b',
        'PhiForCausalLM': 'microsoft/phi-2',
        'OPTForCausalLM': 'facebook/opt-350m',
        'QWenLMHeadModel': 'Qwen/Qwen-7B',
        'QWenForCausalLM': 'Qwen/Qwen-7B',
        'Qwen2ForCausalLM': 'Qwen/Qwen1.5-7B',
        'Qwen2MoeForCausalLM': 'Qwen/Qwen1.5-MoE-A2.7B',
        'RecurrentGemmaForCausalLM': 'google/recurrentgemma-2b',
    }

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

    @staticmethod
    def add_common_args(parser):
        # sampling arguments
        parser.add_argument('--num_beams',
                            type=int,
                            help="Use beam search if num_beams > 1",
                            default=1)
        parser.add_argument('--num_return_sequences',
                            type=int,
                            help="Number of sequences to generate for each input.",
                            default=None)
        parser.add_argument('--temperature', type=float, default=1.0)
        parser.add_argument('--top_k', type=int, default=1)
        parser.add_argument('--top_p', type=float, default=0.0)
        parser.add_argument('--length_penalty', type=float, default=1.0)
        parser.add_argument('--repetition_penalty', type=float, default=1.0)
        parser.add_argument('--presence_penalty', type=float, default=0.0)
        parser.add_argument('--frequency_penalty', type=float, default=0.0)
        parser.add_argument('--beam_search_diversity_rate', type=float, default=0.0)
        parser.add_argument('--random_seed', type=int, default=0)
        parser.add_argument('--early_stopping',
                            type=int,
                            help='Use early stopping if num_beams > 1, '
                                 '1 for early-stopping, 0 for non-early-stopping'
                                 'other values for stopping by length',
                            default=1)
        parser.add_argument(
            '--end_id',
            default=None,
            type=int,
            help="Override tokenizer end_id to stop on given end_id token.")
        parser.add_argument(
            '--stop_words',
            default=None,
            type=str,
            nargs="+",
            action='append',
            help=
            'Set stop words for a batch. Successive invocations of --stop_words set stop words for other batches.'
            '    E.g.: --stop_words " London" " chef" --stop_words "eventually became" "was not"',
        )
        parser.add_argument(
            '--bad_words',
            default=None,
            type=str,
            nargs="+",
            action='append',
            help=
            'Set bad words for a batch. Successive invocations of --bad_words set bad words for other batches.'
            '    E.g.: --bad_words " London" " chef" --bad_words "eventually became" "was not"',
        )
        parser.add_argument('--no_repeat_ngram_size', type=int, default=None)

        # common runtime arguments
        parser.add_argument('--sink_token_length',
                            type=int,
                            default=None,
                            help='The sink token length.')
        parser.add_argument(
            '--max_attention_window_size',
            type=int,
            default=None,
            nargs="+",
            help=
            'The attention window size that controls the sliding window attention / cyclic kv cache behavior'
        )
        parser.add_argument(
            '--multi_block_mode',
            type=lambda s: s.lower() in
                           ("yes", "true", "t", "1"
                            ),  # custom boolean function to convert input string to boolean
            default=True,
            help=
            "Distribute the work across multiple CUDA thread-blocks on the GPU for masked MHA kernel."
        )
        parser.add_argument('--enable_context_fmha_fp32_acc',
                            action='store_true',
                            help="Enable FMHA runner FP32 accumulation.")
        parser.add_argument('--cuda_graph_mode',
                            action='store_true',
                            help="Enable cuda graphs in the inference.")
        parser.add_argument(
            '--log_level',
            type=str,
            choices=['verbose', 'info', 'warning', 'error', 'internal_error'],
            default='info')
        parser.add_argument(
            '--no_prompt_template',
            dest='use_prompt_template',
            default=True,
            action='store_false',
            help=
            "Whether or not to use default prompt template to wrap the input text.")
        parser.add_argument('--use_py_session',
                            default=False,
                            action='store_true',
                            help="Whether or not to use Python runtime session")
        parser.add_argument('--debug_mode',
                            default=False,
                            action='store_true',
                            help="Whether or not to turn on the debug mode")
        parser.add_argument('--streaming', default=False, action='store_true')
        parser.add_argument('--streaming_interval',
                            type=int,
                            help="How often to return tokens when streaming.",
                            default=5)
        parser.add_argument(
            '--prompt_table_path',
            type=str,
            help="Path to .npy file, exported by nemo_prompt_convert.py")
        parser.add_argument(
            '--prompt_tasks',
            help="Comma-separated list of tasks for prompt tuning, e.g., 0,3,1,0")
        parser.add_argument('--lora_dir',
                            type=str,
                            default=None,
                            nargs="+",
                            help="The directory of LoRA weights")
        parser.add_argument('--lora_ckpt_source',
                            type=str,
                            default="hf",
                            choices=["hf", "nemo"],
                            help="The source of lora checkpoint.")
        parser.add_argument(
            '--lora_task_uids',
            type=str,
            default=None,
            nargs="+",
            help="The list of LoRA task uids; use -1 to disable the LoRA module")
        parser.add_argument(
            '--num_prepend_vtokens',
            nargs="+",
            type=int,
            help="Number of (default) virtual tokens to prepend to each sentence."
                 " For example, '--num_prepend_vtokens=10' will prepend the tokens"
                 " [vocab_size, vocab_size + 1, ..., vocab_size + 9] to the sentence.")
        parser.add_argument(
            '--draft_target_model_config',
            type=str,
            default=None,
            help=
            "Configuration of Draft-Target-Model decoding, see `examples/draft_target_model/README.md` for more information."
            "   E.g.: [4, [0], [1], False] for [draft_len, draft_model_device_list, target_model_device_list, use_logits]."
        )
        parser.add_argument(
            '--medusa_choices',
            type=str,
            default=None,
            help="Configuration of Medusa decoding."
                 "   E.g.: [[0, 0, 0, 0], [0, 1, 0], [1, 0], [1, 1]] for 9 medusa tokens."
        )
        parser.add_argument(
            '--lookahead_config',
            type=str,
            default=None,
            help="Configuration of executor and request lookahead decoding."
                 "   E.g.: [5, 6, 7] for [max_window_size, max_ngram_size, max_verification_set_size]."
        )
        # model arguments
        parser.add_argument('--engine_dir', type=str, default='engine_outputs')
        parser.add_argument(
            '--tokenizer_type',
            help=
            'Specify that argument when providing a .model file as the tokenizer_dir. '
            'It allows AutoTokenizer to instantiate the correct tokenizer type.')
        parser.add_argument('--vocab_file',
                            help="Used for sentencepiece tokenizers")
        parser.add_argument('--no_add_special_tokens',
                            dest='add_special_tokens',
                            default=True,
                            action='store_false',
                            help="Whether or not to add special tokens")
        parser.add_argument('--hf_model_dir', '--model_dir', type=str, default=None)
        parser.add_argument(
            '--tokenizer_dir',
            default=None,
            help='tokenizer path; defaults to hf_model_dir if left unspecified')

        # memory argument
        parser.add_argument(
            '--gpu_weights_percent',
            default=1,
            type=float,
            help=
            'Specify the percentage of weights that reside on GPU instead of CPU and streaming load during runtime.',
        )
        parser.add_argument(
            '--max_tokens_in_paged_kv_cache',
            default=None,
            type=int,
            help=
            'Specify the maximum number of tokens in a kv cache page (only available with cpp session).',
        )
        parser.add_argument(
            '--kv_cache_enable_block_reuse',
            action='store_true',
            help=
            'Enables block reuse in kv cache (only available with cpp session).',
        )
        parser.add_argument(
            '--kv_cache_free_gpu_memory_fraction',
            default=0.9,
            type=float,
            help='Specify the free gpu memory fraction.',
        )
        parser.add_argument(
            '--cross_kv_cache_fraction',
            default=0.5,
            type=float,
            help=
            'Specify the kv cache fraction reserved for cross attention. Only applicable for encoder-decoder models. By default 0.5 for self and 0.5 for cross.',
        )
        parser.add_argument(
            '--enable_chunked_context',
            action='store_true',
            help='Enables chunked context (only available with cpp session).',
        )

        # hf model argument (if use hf model)
        parser.add_argument(
            '--hf_data_type',
            '--data_type',
            type=str,
            choices=['fp32', 'fp16', 'bf16', 'float32', 'float16', 'bfloat16'],
            default='fp16',
            help="The data type for hf model.")
        parser.add_argument(
            '--hf_device_map_auto',
            action='store_true',
            help="Use device map 'auto' to load a pretrained HF model. This may "
                 "help to test a large model that cannot fit into a singlue GPU.")

        parser.add_argument(
            "--return_all_generated_tokens",
            default=False,
            action="store_true",
            help="This option changes the token output only for streaming. "
                 "If not specified, return only generated tokens at each step. "
                 "If specified, return the full beams/outputs at each step. "
                 "It is automatically enabled for num_beams>1 (only available with cpp session). "
                 "WARNING: using this option may increase network usage significantly (quadratically w.r.t output length)."
        )

        return parser
