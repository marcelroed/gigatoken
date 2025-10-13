from typing import Optional, List, Tuple
from dataclasses import dataclass
import subprocess
import shlex
from datetime import datetime
from pathlib import Path
import os
import time
import json
from typing import Iterator

from tqdm import tqdm
import submitit
from submitit.core.core import Job
import tiktoken
from tiktoken.load import load_tiktoken_bpe
from copy import copy

import numpy as np
from itertools import chain


@dataclass
class ResourceConfig:
    """
    Dataclass for defining the resources that worker uses
    """

    log_dir: str
    # Slurm
    account: str = "nlp"
    partition: str = "sphinx"
    gres: str = "gpu:0"
    mem: str = "128G"
    time: str = "12:00:00"
    cpus_per_task: int = 8
    exclude: str = ""
    constraints: Optional[str] = None
    # Parallelism
    jobs_per_node: int = 1  # If job can be split into parallel jobs on the cluster side
    # ^ Required for distributed training, set this to number of gpus per node
    # Environment variables
    node_list: Optional[str] = None
    exclusive: bool = False


# ---------------------------------
# Submitit utils
# ---------------------------------


def time_string_to_minutes(time_str: str) -> int:
    """
    Convert time string in format 'DD-HH:MM:SS' or 'HH:MM:SS' to minutes.

    We use this for submitit because it doesn't like DD-HH:MM:SS

    Args:
        time_str: Time string in format 'DD-HH:MM:SS' or 'HH:MM:SS'
    Returns:
        Total minutes as integer
    """
    if not time_str:
        return 0

    # Handle DD-HH:MM:SS format
    if "-" in time_str:
        days_part, time_part = time_str.split("-", 1)
        days = int(days_part)
    else:
        days = 0
        time_part = time_str

    # Parse HH:MM:SS
    time_components = time_part.split(":")
    if len(time_components) == 3:
        hours, minutes, _ = map(int, time_components)
    elif len(time_components) == 2:
        hours, minutes = map(int, time_components)
    else:
        raise ValueError(f"Invalid time format: {time_str}")

    # Convert to total minutes
    total_minutes = days * 24 * 60 + hours * 60 + minutes

    return total_minutes


def get_submitit_executor(
    resources_config: ResourceConfig,
    log_dir: Optional[str] = None,
) -> submitit.AutoExecutor:
    # Get the path that Python is being run from
    # calling_dir = os.getcwd()

    # convert time to minutes as submitit premting doesn't support time strings
    time = time_string_to_minutes(resources_config.time)

    if log_dir is None:
        log_dir = resources_config.log_dir

    for k in list(os.environ):
        if k.startswith("SLURM_"):
            os.environ.pop(k, None)
    os.environ["SLURM_CPU_BIND"] = "none"  # set a safe default for the new job

    executor = submitit.AutoExecutor(folder=log_dir)
    executor.update_parameters(
        slurm_account=resources_config.account,
        slurm_partition=resources_config.partition,
        slurm_gres=resources_config.gres,
        slurm_mem=resources_config.mem,
        slurm_time=time,
        slurm_cpus_per_task=resources_config.cpus_per_task,
        slurm_exclude=resources_config.exclude,
        slurm_ntasks_per_node=resources_config.jobs_per_node,
        slurm_constraint=resources_config.constraints,
        slurm_additional_parameters={
            "export": "ALL,SLURM_CPU_BIND=none",
            # "threads-per-core": "1"
        },
    )

    if resources_config.node_list is not None:
        executor.update_parameters(
            slurm_nodelist=resources_config.node_list,
        )

    if resources_config.exclusive:
        executor.update_parameters(
            slurm_exclusive=True,
        )

    return executor

def cleanup_submitit_job(job: Job):
    """
    Job will be cancelled and associated files destroyed. If Job is already cancelled then nothing
    will happen.

    This means after running log files will be destroyed.
    """

    path: Path
    for path in [
        job.paths.stderr,
        job.paths.stdout,
        job.paths.submission_file,
        job.paths.submitted_pickle,
        job.paths.result_pickle,
    ]:
        if path.exists():
            path.unlink()


# -------------------------------------------------
# Tokenization driver
# -------------------------------------------------


def load_tiktoken_tokenizer(tokenizer_path: str) -> tiktoken.Encoding:
    """
    Adapted from https://github.com/tatsu-lab/lingua-fork/blob/57c6a022fd220e0e9dea4facaf5075ffb5931704/lingua/tokenizer.py#L194
    """

    DEFAULT_TIKTOKEN_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
    DEFAULT_TIKTOKEN_SPECIAL_TOKENS = {
        "<|begin_of_text|>": 0,
        "<|end_of_text|>": 1,
        "<|fim_prefix|>": 2,
        "<|fim_middle|>": 3,
        "<|fim_end_fill|>": 253,
        "<|fim_pad|>": 254,
        "<|fim_suffix|>": 255,
    }
    TIKTOKEN_MAX_ENCODE_CHARS = 400_000


    mergeable_ranks = load_tiktoken_bpe(tokenizer_path)
    all_special_tokens_with_ids = copy(DEFAULT_TIKTOKEN_SPECIAL_TOKENS)
    missing_ids = set(range(256)) - set(all_special_tokens_with_ids.values())
    for id in missing_ids:
        all_special_tokens_with_ids[f"<|reserved_special_token_{id}|>"] = id
    for name in all_special_tokens_with_ids:
        all_special_tokens_with_ids[name] += len(mergeable_ranks)

    tkt_model = tiktoken.core.Encoding(
        name=Path(tokenizer_path).stem,
        pat_str=DEFAULT_TIKTOKEN_PATTERN,
        mergeable_ranks=mergeable_ranks,
        special_tokens=all_special_tokens_with_ids,
    )

    return tkt_model

def tokenize_chunk(
        docs: List[str], 
        tiktoken_tokenizer: tiktoken.Encoding,
        num_threads: int,
        add_bos: bool,
        add_eos: bool,
) -> List[List[int]]:

        bos_id: int = tiktoken_tokenizer.encode_single_token("<|begin_of_text|>")
        eos_id: int = tiktoken_tokenizer.encode_single_token("<|end_of_text|>")

        # For computing bytes per token
        num_input_bytes = sum(len(doc) for doc in docs)

        tokenized_docs: List[List[int]] = tiktoken_tokenizer.encode_batch(docs, num_threads=num_threads)        

        tokenized_docs = [
            [bos_id] * add_bos + doc + [eos_id] * add_eos
            for doc in tokenized_docs
        ]

        num_tokens = sum(len(toks) for toks in tokenized_docs)

        return tokenized_docs, num_input_bytes, num_tokens
        

def tokenize_dataset(
    jsonl_paths: List[Path],
    document_key: str,
    save_directory: Path,
    tokenizer_path: str,
    num_threads: int,
    chunk_size: int = 2_000_000,
) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    save_directory.mkdir(parents=True, exist_ok=True)
    token_dtype = np.uint32

    tokenizer = load_tiktoken_tokenizer(tokenizer_path)

    def iter_jsonl_chunks(p: Path, key: str, n: int) -> Iterator[Tuple[List[str], List[dict], int, float]]:

        docs, rows = [], []
        bytes_read = 0
        fails = 0
        chunk_start = time.time()
        with open(p, "rb") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                try:
                    bytes_read += len(line)
                    data = json.loads(line)  # bytes are fine for json.loads
                    rows.append(data)
                    docs.append(data[key])
                    if len(docs) >= n:
                        load_time = time.time() - chunk_start
                        yield docs, rows, bytes_read, load_time
                        docs, rows, bytes_read = [], [], 0
                        chunk_start = time.time()
                except json.JSONDecodeError:
                    fails += 1

        if docs:
            load_time = time.time() - chunk_start
            yield docs, rows, bytes_read, load_time
        print(f"Failed to parse {fails} lines in {p}")

    print("=" * 100)
    print("Starting tokenization")
    print("=" * 100)

    # We process each jsonl file one at a time
    for jsonl_path in tqdm(jsonl_paths, desc="Files", position=0):
        file_size = os.path.getsize(jsonl_path)

        metadata_path = save_directory / f"{jsonl_path.stem}_metadata.json"
        memmap_path = save_directory / f"{jsonl_path.stem}_tokens.np"
        # Fresh run: truncate existing memmap if present
        if memmap_path.exists():
            memmap_path.unlink()
        offset = 0

        # Note we cannot just use the filesize as this will include metadata
        total_bytes_in_docs = 0
        total_tokens = 0


        with (
            tqdm(
                total=file_size,
                desc=f"Tokenizing {jsonl_path.name}",
                position=1,
                leave=False,
                dynamic_ncols=True,
                unit="B",
                unit_scale=True,
            ) as pbar,
        ):
            for docs, _, bytes_in_chunk, load_time in iter_jsonl_chunks(
                jsonl_path, document_key, chunk_size
            ):

                chunk_start = time.time()
                print(f"Load chunk time: {load_time:.2f} seconds")

                start = time.time()

                tokenized_docs: List[List[int]]

                tokenized_docs, num_input_bytes, num_tokens = tokenize_chunk(
                    docs=docs,
                    tiktoken_tokenizer=tokenizer,
                    num_threads=num_threads,
                    add_bos=True,
                    add_eos=True,
                )

                total_bytes_in_docs += num_input_bytes
                total_tokens += num_tokens

                tokenize_time = time.time() - start
                print(f"Tokenized chunk time: {tokenize_time:.2f} seconds")


                start = time.time()
                # Flatten and append to memmap
                flat_chunk = np.fromiter(
                    chain.from_iterable(tokenized_docs),
                    dtype=token_dtype,
                )
                if flat_chunk.size:
                    new_total = offset + flat_chunk.size
                    # Ensure file large enough, then write into slice
                    with open(memmap_path, "a+b") as f:
                        f.truncate(new_total * np.dtype(token_dtype).itemsize)
                    mm = np.memmap(
                        memmap_path,
                        dtype=token_dtype,
                        mode="r+",
                        shape=(new_total,),
                    )
                    mm[offset:new_total] = flat_chunk
                    mm.flush()
                    del mm
                    offset = new_total

                write_time = time.time() - start
                print(f"Write chunk time: {write_time:.2f} seconds")

                chunk_end = time.time()
                mbs = bytes_in_chunk / 1024 / 1024
                print(f"Chunk time: {chunk_end - chunk_start:.2f} seconds")
                print(f"MB processed: {mbs:.2f}")
                print(f"MB/s processed: {mbs / (chunk_end - chunk_start):.2f}")
                print("=" * 100)
                print()

                pbar.update(bytes_in_chunk)

        # For now we save this simply metadata, in the future we can save more
        with open(metadata_path, "w") as f:
            json.dump({
                "total_bytes_in_docs": total_bytes_in_docs,
                "total_tokens": total_tokens,
                "tokenizer_path": tokenizer_path,
                "untokenized_data_path": str(jsonl_path),
            }, f, indent=4)

    return None


def combined_mmemaps(
    save_path: Path,
    metadata_paths: List[Path],
    memmap_paths: List[Path],
) -> None:
    """
    Concatenate many shard memmaps into one giant memmap using kernel-space copy.

    Assumptions:
    - os.posix_fallocate is available.
    - os.copy_file_range is available.
    - Shards live on the same filesystem for maximum throughput.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    token_dtype = np.uint32
    itemsize = np.dtype(token_dtype).itemsize

    shard_sizes = [os.path.getsize(p) for p in memmap_paths]
    total_bytes = sum(shard_sizes)
    tokens_per_shard = [sz // itemsize for sz in shard_sizes]
    total_tokens = sum(tokens_per_shard)

    print("=" * 100)
    print(f"Combining {len(memmap_paths)} shards into {save_path}")
    print(f"Total size: {total_bytes / (1024**3):.2f} GiB; expected tokens: {total_tokens:,}")
    print("=" * 100)

    start = time.time()
    pbar = tqdm(
        total=total_bytes,
        desc="Combining memmaps",
        unit="B",
        unit_scale=True,
        dynamic_ncols=True,
        leave=False,
    )

    dst_fd = os.open(save_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.posix_fallocate(dst_fd, 0, total_bytes)

    max_chunk = 1 << 30  # 1 GiB chunk to avoid kernel limits
    for src_path, sz in zip(memmap_paths, shard_sizes):
        src_fd = os.open(src_path, os.O_RDONLY)
        remaining = sz
        while remaining:
            to_copy = min(remaining, max_chunk)
            n = os.copy_file_range(src_fd, dst_fd, to_copy)
            if n == 0:
                break
            remaining -= n
            pbar.update(n)
        os.close(src_fd)

    os.fsync(dst_fd)
    os.close(dst_fd)
    pbar.close()

    # Aggregate and write combined metadata
    combined_metadata_path = save_path.with_name(save_path.stem + "_metadata.json")
    total_bytes_in_docs = 0
    total_tokens_meta = 0
    tokenizer_paths = set()
    for mp in metadata_paths:
        with open(mp) as f:
            m = json.load(f)
        total_bytes_in_docs += m.get("total_bytes_in_docs", 0)
        total_tokens_meta += m.get("total_tokens", 0)
        tp = m.get("tokenizer_path")
        if tp:
            tokenizer_paths.add(tp)

    out_meta = {
        "total_tokens": int(total_tokens),
        "token_dtype": "uint32",
        "component_memmaps": [str(p) for p in memmap_paths],
        "component_metadata": [str(p) for p in metadata_paths],
        "total_bytes_in_docs": int(total_bytes_in_docs),
        "source_total_tokens_sum": int(total_tokens_meta),
        "created_at": datetime.now().isoformat(),
    }
    if tokenizer_paths:
        out_meta["tokenizer_path"] = list(tokenizer_paths)[0] if len(tokenizer_paths) == 1 else list(tokenizer_paths)

    with open(combined_metadata_path, "w") as f:
        json.dump(out_meta, f, indent=4)

    elapsed = time.time() - start
    mibs = total_bytes / (1024 * 1024)
    print("=" * 100)
    print(f"Combined -> {save_path}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Elapsed: {elapsed:.2f}s; Throughput: {mibs / elapsed:.2f} MiB/s")
    print("=" * 100)




# -------------------------------------------------
# Test
# -------------------------------------------------

if __name__ == "__main__":

    # This sets off jobs to tokenize all of fineweb_edu.
    # Assuming good cluster availability, it should take less than 12 hours to tokenize the entire thing.

    tokenizer_path = "/juice5/scr5/nlp/data/huggingface/lingua-data/tokenizers/r50k_base_tokenizer/0ea1e91bbb3a60f729a8dc8f777fd2fc07cd8df4"
    #jsonl_stub = "/juice5/scr5/nlp/data/huggingface/lingua-data/fineweb_edu/fineweb_edu.chunk."
    jsonl_stub = "/juice5/scr5/nlp/data/huggingface/lingua-data/dclm_baseline_1_0_shuffled/dclm_baseline_1.0.chunk."
    #save_dir = Path("/juice5b/scr5b/nlp/data/huggingface/lingua-data/elastic_lm/fineweb_edu_tokenized")
    save_dir = Path("/juice5/scr5/nlp/data/huggingface/lingua-data/elastic_lm/dclm_baseline_tokenized")

    master_resource_config = ResourceConfig(
        log_dir="./logs",
        # Slurm
        account="nlp",
        partition="jag-standard,sphinx",
        gres="gpu:0",
        time="2-00:00:00",
        mem="64G",
        cpus_per_task=16,
    )

    executor = get_submitit_executor(master_resource_config)

    for i in range(16):
        #jsonl_path = jsonl_stub + str(i).zfill(5) + ".jsonl"
        jsonl_path = jsonl_stub + str(i).zfill(2) + ".jsonl"
        jsonl_paths = [Path(jsonl_path)]

        executor.submit(
            tokenize_dataset,
            jsonl_paths=jsonl_paths,
            document_key="text",
            save_directory=save_dir,
            tokenizer_path=tokenizer_path,
            chunk_size=100_000,
            num_threads=master_resource_config.cpus_per_task,
        )

    # NOTE UNTESTED FOR NOW

    #mmemap_stubs = "/nlp/scr5/nlp/data/huggingface/lingua-data/elastic_lm/fineweb_edu_tokenized/fineweb_edu.chunk."
    #shard_paths = []
    #metadata_paths = []
    #for i in range(64):
    #    shard_paths.append(Path(mmemap_stubs + str(i).zfill(5) + "_tokens.np"))
    #    metadata_paths.append(Path(mmemap_stubs + str(i).zfill(5) + "_metadata.json"))

    #combined_mmemaps(
    #    save_path=Path("/juice5/scr5/nlp/data/huggingface/lingua-data/elastic_lm/fineweb_edu_tokenized/fineweb_edu_tokenized_all.np"),
    #    metadata_paths=metadata_paths,
    #    memmap_paths=shard_paths,
    #)