# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import IterableDataset, Dataset, load_dataset, interleave_datasets # huggingface datasets
from functools import partial
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 4

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")

def get_mixed_data():
    ds1 = load_dataset("codelion/finepdfs-1B", split="train", streaming=True)
    ds2 = load_dataset("codelion/dclm-baseline-1B", split="train", streaming=True)
    ds3 = load_dataset("codelion/fineweb-edu-1B", split="train", streaming=True)

    def estimate_avg_len(ds, num=1000):
        lens = []
        it = iter(ds)
        for _ in range(num):
            try:
                lens.append(len(next(it)["text"].split()))
            except StopIteration:
                break
        return np.mean(lens) if lens else 0
    
    avg1 = estimate_avg_len(ds1)
    avg2 = estimate_avg_len(ds2)
    avg3 = estimate_avg_len(ds3)

    ds1 = load_dataset("codelion/finepdfs-1B", split="train", streaming=True)
    ds2 = load_dataset("codelion/dclm-baseline-1B", split="train", streaming=True)
    ds3 = load_dataset("codelion/fineweb-edu-1B", split="train", streaming=True)

    p_tokens = [0.5, 0.3, 0.2]
    L = [avg1, avg2, avg3]
    raw = [p_tokens[i] / L[i] for i in range(3)]
    probs = [r / sum(raw) for r in raw]

    mixed: IterableDataset = interleave_datasets(
        [ds1, ds2, ds3],
        probabilities=probs,
        seed=42,
        stopping_strategy="all_exhausted"
    )
    def mixed_generator(iter_ds):
        yield from iter_ds
    mixed = Dataset.from_generator(partial(mixed_generator, mixed), features=mixed.features)
    return mixed


if __name__ == '__main__':
    dataset = get_mixed_data()

    # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
    def process(example):
        ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {'ids': ids, 'len': len(ids)}
        return out

    # tokenize the dataset
    tokenized = dataset.map(
        process,
        remove_columns=['text'],
    )

    print(dataset) # should have 'ids' and 'len' fields, and 'ids' should be a list of token ids
    print(tokenized) # should have 'ids' and 'len' fields, and 'ids' should be a list of token ids

    # concatenate all the ids in each dataset into one large file we can use for training
    arr_len = np.sum(tokenized['len'], dtype=np.uint64)
    print(f'total tokens: {arr_len:,}') # should be ~1B for train
    filename = os.path.join(os.path.dirname(__file__), 'train.bin')
    dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    total_batches = 1024

    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
        # Batch together samples for faster write
        batch = tokenized.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
        arr_batch = np.concatenate(batch['ids'])
        # Write into mmap
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
    arr.flush()

    # create empty val.bin for now, we can fill it later if we want
    arr_len = 0
    filename = os.path.join(os.path.dirname(__file__), 'val.bin')
    dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    arr.flush()

    # train.bin is ~17GB, val.bin ~8.5MB
    # train has ~9B tokens (9,035,582,198)
    # val has ~4M tokens (4,434,897)

    # to read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')
