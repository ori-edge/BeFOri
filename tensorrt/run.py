import argparse
import ast
import csv

import numpy as np

import tensorrt_llm
import tensorrt_llm.profiler
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner
import json
import os
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoTokenizer, LlamaTokenizer, T5Tokenizer

from tensorrt_llm._utils import supports_inflight_batching  # noqa
from tensorrt_llm.builder import get_engine_version

def parse_arguments(args=None):
    # see `add_common_args` for extended list of arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_input_length', type=int, default=923)
    parser.add_argument('--max_output_len', type=int, required=True)
    parser.add_argument(
        '--draft_engine_dir',
        type=str,
        default=None,
        help='Path to engine of draft model in Draft-Target-Model mode.')
    parser.add_argument(
        '--input_text',
        type=str,
        nargs='+',
        default=["Born in north-east France, Soyer trained as a"])
    parser.add_argument(
        '--input_file',
        type=str,
        help=
        'CSV or Numpy file containing tokenized input. Alternative to text input.',
        default=None)
    parser.add_argument('--multimodal_input_file',
                        type=str,
                        help='Path to multimodal input file.')
    parser.add_argument(
        '--input_token_extra_ids',
        type=int,
        nargs='+',
        help=
        'Input token extra ids for using p-tuning and KV Cache reuse together (only available with cpp session).',
        default=None)
    parser.add_argument(
        '--input_token_extra_ids_file',
        type=str,
        help=
        'CSV or Numpy file containing input token extra ids file. Alternative to text input (only available with cpp session).',
        default=None)
    parser.add_argument('--output_csv',
                        type=str,
                        help='CSV file where the tokenized output is stored.',
                        default=None)
    parser.add_argument('--output_npy',
                        type=str,
                        help='Numpy file where the tokenized output is stored.',
                        default=None)
    parser.add_argument(
        '--output_logits_npy',
        type=str,
        help=
        'Numpy file where the generation logits are stored. Use only when num_beams==1',
        default=None)
    parser.add_argument('--output_log_probs_npy',
                        type=str,
                        help='Numpy file where the log_probs are stored',
                        default=None)
    parser.add_argument('--output_cum_log_probs_npy',
                        type=str,
                        help='Numpy file where the cum_log_probs are stored',
                        default=None)
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

    parser = add_common_args(parser)

    return parser.parse_args(args=args)

#KEEP
def parse_input(tokenizer,
                input_text=None,
                input_file=None,
                add_special_tokens=True,
                max_input_length=923,
                pad_id=None,
                num_prepend_vtokens=[],
                model_name=None,
                model_version=None):
    if pad_id is None:
        pad_id = tokenizer.pad_token_id

    batch_input_ids = []
    if input_file is None:
        if 'whisper' in model_name.lower():
            batch_input_ids.append(tokenizer.prefix_tokens)
        else:
            for curr_text in input_text:
                input_ids = tokenizer.encode(
                    curr_text,
                    add_special_tokens=add_special_tokens,
                    truncation=True,
                    max_length=max_input_length)
                batch_input_ids.append(input_ids)
    else:
        if input_file.endswith('.csv'):
            with open(input_file, 'r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for line in csv_reader:
                    input_ids = np.array(line, dtype='int32')
                    batch_input_ids.append(input_ids[-max_input_length:])
        elif input_file.endswith('.npy'):
            inputs = np.load(input_file)
            for row in inputs:
                input_ids = row[row != pad_id]
                batch_input_ids.append(input_ids[-max_input_length:])
        elif input_file.endswith('.txt'):
            with open(input_file, 'r', encoding='utf-8',
                      errors='replace') as txt_file:
                input_text = txt_file.readlines()
                batch_input_ids = tokenizer(
                    input_text,
                    add_special_tokens=add_special_tokens,
                    truncation=True,
                    max_length=max_input_length)["input_ids"]
        else:
            print('Input file format not supported.')
            raise SystemExit

    if num_prepend_vtokens:
        assert len(num_prepend_vtokens) == len(batch_input_ids)
        base_vocab_size = tokenizer.vocab_size - len(
            tokenizer.special_tokens_map.get('additional_special_tokens', []))
        for i, length in enumerate(num_prepend_vtokens):
            batch_input_ids[i] = list(
                range(base_vocab_size,
                      base_vocab_size + length)) + batch_input_ids[i]

    if input_file is None and 'GLM' in model_name and model_version == 'glm':
        for ids in batch_input_ids:
            ids.append(tokenizer.sop_token_id)

    batch_input_ids = [
        torch.tensor(x, dtype=torch.int32) for x in batch_input_ids
    ]

    logger.debug(f"Input token ids (batch_size = {len(batch_input_ids)}):")
    for i, input_ids in enumerate(batch_input_ids):
        logger.debug(f"Request {i}: {input_ids.tolist()}")

    return batch_input_ids

#KEEP
def parse_input_token_extra_ids(prompt_table_path, kv_cache_enable_block_reuse,
                                input_token_extra_ids,
                                input_token_extra_ids_file, max_input_length):
    batch_extra_ids = None
    if prompt_table_path and kv_cache_enable_block_reuse:
        assert input_token_extra_ids or input_token_extra_ids_file, \
            "Input token extra ids must be provided when p-tuning and KV Cache reuse are both enabled"
        batch_extra_ids = []
        if input_token_extra_ids_file:
            if input_token_extra_ids_file.endswith('.csv'):
                with open(input_token_extra_ids_file, 'r') as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',')
                    for line in csv_reader:
                        extra_ids = [int(num) for num in line]
                        batch_extra_ids.append(extra_ids[-max_input_length:])
            elif input_token_extra_ids_file.endswith('.npy'):
                inputs = np.load(input_token_extra_ids_file)
                for extra_ids in inputs:
                    batch_extra_ids.append(extra_ids[-max_input_length:])
            else:
                print('Input file format not supported.')
                raise SystemExit
        else:
            batch_extra_ids.append(input_token_extra_ids)
    return batch_extra_ids


def load_tokenizer(model_name: Optional[str], tokenizer_dir: str = "../output/"):
    # Load tokenizer if it's in the tokenizer_dir
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    except OSError as e:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ.get("HF_ACCESS_TOKEN"), save_pretrained=tokenizer_dir)
    tokenizer.add_special_tokens({'pad_token': '<|reserved_special_token_0|>'})
    tokenizer.pad_token_id = 128002
    pad_id = tokenizer.pad_token_id
    end_id = tokenizer.eos_token_id
    return tokenizer, pad_id, end_id


def read_model_name(engine_dir: str):
    engine_version = get_engine_version(engine_dir)

    with open(Path(engine_dir) / "config.json", 'r') as f:
        config = json.load(f)

    if engine_version is None:
        return config['builder_config']['name'], None

    model_arch = config['pretrained_config']['architecture']
    model_version = None
    if 'GLM' in model_arch:
        model_version = config['pretrained_config']['chatglm_version']
    if 'qwen' in model_arch.lower():
        model_version = config['pretrained_config']['qwen_type']
    return model_arch, model_version


#KEEP
def print_output(tokenizer,
                 output_ids: torch.Tensor,
                 input_lengths: List[int],
                 sequence_lengths: torch.Tensor,
                 output_csv: Optional[str] = None,
                 output_npy: Optional[str] = None,
                 context_logits: Optional[torch.Tensor] = None,
                 generation_logits: Optional[torch.Tensor] = None,
                 cum_log_probs: Optional[torch.Tensor] = None,
                 log_probs: Optional[torch.Tensor] = None,
                 output_logits_npy: Optional[str] = None,
                 output_cum_log_probs_npy: Optional[str] = None,
                 output_log_probs_npy: Optional[str] = None):
    num_output_sents, num_beams, _ = output_ids.size()
    batch_size = len(input_lengths)
    num_return_sequences = num_output_sents // batch_size

    if output_csv is None and output_npy is None:
        for i in range(batch_size * num_return_sequences):
            batch_idx = i // num_return_sequences
            seq_idx = i % num_return_sequences
            inputs = output_ids[i][0][:input_lengths[batch_idx]].tolist()
            input_text = tokenizer.decode(inputs)
            if seq_idx == 0:
                print(f'Input [Text {batch_idx}]: \"{input_text}\"')

            for beam in range(num_beams):
                output_begin = input_lengths[batch_idx]
                output_end = sequence_lengths[i][beam]
                outputs = output_ids[i][beam][output_begin:output_end].tolist()
                output_text = tokenizer.decode(outputs)
                index_str = (f'Text {batch_idx} Seq {seq_idx} Beam {beam}'
                             if num_return_sequences > 1 else
                             f'Text {batch_idx} Beam {beam}')
                print(f'Output [{index_str}]: \"{output_text}\"')
                logger.debug(str(outputs))

    output_ids = output_ids.reshape((-1, output_ids.size(2)))

    if output_csv is not None:
        output_file = Path(output_csv)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        outputs = output_ids.tolist()
        with open(output_file, 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerows(outputs)

    if output_npy is not None:
        output_file = Path(output_npy)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        outputs = np.array(output_ids.cpu().contiguous(), dtype='int32')
        np.save(output_file, outputs)

    # Save context logits
    if context_logits is not None and output_logits_npy is not None:
        context_logits = torch.cat(context_logits, axis=0)
        vocab_size_padded = context_logits.shape[-1]
        context_logits = context_logits.reshape([1, -1, vocab_size_padded])

        output_context_logits_npy = output_logits_npy.split(
            '.npy')[0] + "_context"
        output_context_logits_file = Path(output_context_logits_npy)
        context_outputs = np.array(
            context_logits.squeeze(0).cpu().contiguous(),
            dtype='float32')  # [promptLengthSum, vocabSize]
        np.save(output_context_logits_file, context_outputs)

    # Save generation logits
    if generation_logits is not None and output_logits_npy is not None and num_beams == 1:
        output_generation_logits_npy = output_logits_npy.split(
            '.npy')[0] + "_generation"
        output_generation_logits_file = Path(output_generation_logits_npy)
        generation_outputs = np.array(generation_logits.cpu().contiguous(),
                                      dtype='float32')
        np.save(output_generation_logits_file, generation_outputs)

    # Save cum log probs
    if cum_log_probs is not None and output_cum_log_probs_npy is not None:
        cum_log_probs_file = Path(output_cum_log_probs_npy)
        cum_log_probs_outputs = np.array(cum_log_probs.cpu().contiguous(),
                                         dtype='float32')
        np.save(cum_log_probs_file, cum_log_probs_outputs)

    # Save cum log probs
    if log_probs is not None and output_log_probs_npy is not None:
        log_probs_file = Path(output_log_probs_npy)
        log_probs_outputs = np.array(log_probs.cpu().contiguous(),
                                     dtype='float32')
        np.save(log_probs_file, log_probs_outputs)


def throttle_generator(generator, stream_interval):
    for i, out in enumerate(generator):
        if i == 0:
            # Always yield the first token
            yield out
        elif not i % stream_interval:
            yield out
    # Ensure last token(s) are yielded
    if i % stream_interval:
        yield out


#KEEP
def main(args):
    runtime_rank = tensorrt_llm.mpi_rank()
    logger.set_level(args.log_level)

    # Python sessions don't handle encoder-decoder models
    is_enc_dec = {'encoder', 'decoder'}.issubset({
        name
        for name in os.listdir(args.engine_dir)
        if os.path.isdir(os.path.join(args.engine_dir, name))
    })
    assert not is_enc_dec, f"""
    This path {args.engine_dir} is an encoder-decoder model. 
    Encoder-decoder models don't have a unified python runtime, implementation not supported.
    """

    model_name, model_version = read_model_name(args.engine_dir)

    assert args.tokenizer_dir is not None, "Need to specify tokenizer_dir is not specified."

    tokenizer, pad_id, end_id = load_tokenizer(
        tokenizer_dir=args.tokenizer_dir,
        vocab_file=args.vocab_file,
        model_name=model_name,
        model_version=model_version,
        tokenizer_type=args.tokenizer_type,
    )

    if args.end_id:
        end_id = args.end_id

    batch_input_ids = parse_input(tokenizer=tokenizer,
                                  input_text=args.input_text,
                                  input_file=args.input_file,
                                  add_special_tokens=args.add_special_tokens,
                                  max_input_length=args.max_input_length,
                                  pad_id=pad_id,
                                  num_prepend_vtokens=args.num_prepend_vtokens,
                                  model_name=model_name,
                                  model_version=model_version)

    stop_words_list = None
    if args.stop_words:
        stop_words_list = tensorrt_llm.runtime.decode_words_list(
            args.stop_words, tokenizer)
    if model_version == 'glm4':  # add default stop token ids for GLM-4
        glm4_stop_ids = [[151329], [151336], [151338]]
        if stop_words_list is None:
            stop_words_list = [glm4_stop_ids] * len(batch_input_ids)
        else:
            for req_stop_words_list in stop_words_list:
                req_stop_words_list.extend(glm4_stop_ids)

    bad_words_list = None
    if args.bad_words:
        bad_words_list = tensorrt_llm.runtime.decode_words_list(
            args.bad_words, tokenizer)

    input_token_extra_ids = parse_input_token_extra_ids(
        args.prompt_table_path, args.kv_cache_enable_block_reuse,
        args.input_token_extra_ids, args.input_token_extra_ids_file,
        args.max_input_length)

    input_lengths = [x.size(0) for x in batch_input_ids]

    logger.info(f"Using Python session")

    runner_cls = ModelRunner
    runner_kwargs = dict(
        engine_dir=args.engine_dir,
        lora_dir=args.lora_dir,
        rank=runtime_rank,
        debug_mode=args.debug_mode,
        lora_ckpt_source=args.lora_ckpt_source,
        gpu_weights_percent=args.gpu_weights_percent,
        max_output_len=args.max_output_len,
    )
    if args.medusa_choices is not None:
        args.medusa_choices = ast.literal_eval(args.medusa_choices)
        assert args.temperature == 1.0, "Medusa should use temperature == 1.0"
        assert args.num_beams == 1, "Medusa should use num_beams == 1"
        runner_kwargs.update(medusa_choices=args.medusa_choices)
    if args.lookahead_config is not None:
        args.lookahead_config = ast.literal_eval(args.lookahead_config)
        assert len(
            args.lookahead_config
        ) == 3, "Lookahead needs [max_window_size, max_ngram_size, max_verification_set_size]"
        runner_kwargs.update(lookahead_config=args.lookahead_config)

    runner_kwargs.update(
        enable_context_fmha_fp32_acc=args.enable_context_fmha_fp32_acc)
    runner = runner_cls.from_dir(**runner_kwargs)

    with torch.no_grad():
        outputs = runner.generate(
            batch_input_ids=batch_input_ids,
            encoder_input_ids=None,
            encoder_input_features=None,
            encoder_output_lengths=None,
            max_new_tokens=args.max_output_len,
            max_attention_window_size=args.max_attention_window_size,
            sink_token_length=args.sink_token_length,
            end_id=end_id,
            pad_id=pad_id,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_beams=args.num_beams,
            num_return_sequences=args.num_return_sequences,
            length_penalty=args.length_penalty,
            early_stopping=args.early_stopping,
            repetition_penalty=args.repetition_penalty,
            presence_penalty=args.presence_penalty,
            frequency_penalty=args.frequency_penalty,
            stop_words_list=stop_words_list,
            bad_words_list=bad_words_list,
            output_cum_log_probs=(args.output_cum_log_probs_npy != None),
            output_log_probs=(args.output_log_probs_npy != None),
            random_seed=args.random_seed,
            lora_uids=args.lora_task_uids,
            prompt_table=args.prompt_table_path,
            prompt_tasks=args.prompt_tasks,
            streaming=True,
            output_sequence_lengths=True,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            return_dict=True,
            medusa_choices=args.medusa_choices,
            return_all_generated_tokens=False,
            input_token_extra_ids=input_token_extra_ids)
        torch.cuda.synchronize()

    # Receive output, print to screen or save to file
    for curr_outputs in throttle_generator(outputs,
                                           args.streaming_interval):
        if runtime_rank == 0:
            output_ids = curr_outputs['output_ids']
            sequence_lengths = curr_outputs['sequence_lengths']
            cum_log_probs = None
            log_probs = None
            if args.output_cum_log_probs_npy is not None:
                cum_log_probs = curr_outputs['cum_log_probs']
            if args.output_log_probs_npy is not None:
                log_probs = curr_outputs['log_probs']
            print_output(
                tokenizer,
                output_ids,
                input_lengths,
                sequence_lengths,
                output_csv=args.output_csv,
                output_npy=args.output_npy,
                cum_log_probs=cum_log_probs,
                log_probs=log_probs,
                output_cum_log_probs_npy=args.output_cum_log_probs_npy,
                output_log_probs_npy=args.output_log_probs_npy)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
