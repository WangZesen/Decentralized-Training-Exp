import os
import struct
import sys
import time
import tomli_w
import multiprocessing as mp
from typing import Tuple
from src.conf import parse_config, Config, SPECIAL_TOKENS
import tokenizers
from tokenizers.trainers import BpeTrainer, Trainer, WordPieceTrainer
from tokenizers.models import BPE, WordPiece
from tokenizers import Tokenizer
from loguru import logger
logger.remove()
logger.add(sys.stdout)


REMAP_DICT = {
    '„ ' : '"', # fix non-aligned beginnings
    ' “' : '"', # fix non-aligned beginnings
    '\u0093' : '"',
    '\u0094' : '"',
    '\u0097' : ' ',
    ' “' : '"', # fix non-aligned beginnings
    '\u00a0' : ' ', # non-breaking white space
    '\u202f' : ' ', # narrow non-breaking white space
    'Ã¶' : 'ö', # german oe
    'Ã¼' : 'ü', # german ue
    'Ã¤' : 'ä', # german ae
    '„'  : '"',
    '“'  : '"',
    '‟'  : '"',
    '”'  : '"',
    '″'  : '"',
    '‶'  : '"',
    '”'  : '"',
    '‹'  : '"',
    '›'  : '"',
    '’'  : "'",
    '′'  : "'",
    '′'  : "'",
    '‛'  : "'",
    '‘'  : "'",
    '`'  : "'",
    '–'  : '--',
    '‐'  : '-',
    '»'  : '"',
    '«'  : '"',
    '≪'  : '"',
    '≫'  : '"',
    '》' : '"',
    '《' : '"',
    '？' : '?',
    '！' : '!',
    '…'  : ' ... ',
    '\t' : ' ',
    '。' : '.', # chinese period
    '︰' : ':',
    '〜' : '~',
    '；' : ';',
    '）' : ')',
    '（' : '(',
    'ﬂ'  : 'fl', # small ligature characters
    'ﬁ'  : 'fi',
    '¶'  : ' ',
}


def clean_worker(src_file: str, tgt_file: str, idx: int):
    cleaned_src_file = f'{src_file}.{idx:02d}.cleaned'
    cleaned_tgt_file = f'{tgt_file}.{idx:02d}.cleaned'
    src_file = f'{src_file}.{idx:02d}'
    tgt_file = f'{tgt_file}.{idx:02d}'
    

    with open(src_file, 'r') as f_src, open(tgt_file, 'r') as f_tgt:
        full_src = f_src.read()
        full_tgt = f_tgt.read()
    
    nxtline = "\n"
    assert len(full_src.split(nxtline)) == len(full_tgt.split(nxtline)), f"{len(full_src.split(nxtline))} {len(full_tgt.split(nxtline))}"
    
    for old, new in REMAP_DICT.items():
        full_src = full_src.replace(old, new)
        full_tgt = full_tgt.replace(old, new)
    
    while full_src.find('  ') >= 0:
        full_src = full_src.replace('  ', ' ').replace('  ', ' ')
    while full_tgt.find('  ') >= 0:
        full_tgt = full_tgt.replace('  ', ' ').replace('  ', ' ')
    
    full_src = full_src.split(nxtline)
    full_tgt = full_tgt.split(nxtline)

    i = 0
    while i < len(full_src):
        if len(full_src[i]) == 0 or len(full_tgt[i]) == 0:
            full_src.pop(i)
            full_tgt.pop(i)
        else:
            i += 1

    with open(cleaned_src_file, 'w') as f:
        f.write('\n'.join(full_src))
    with open(cleaned_tgt_file, 'w') as f:
        f.write('\n'.join(full_tgt))


def split_to_multi_files(file_dir: str, num_files: int):
    count = 0
    with open(file_dir, 'r') as f:
        line = f.readline()
        while line:
            line = f.readline()
            count += 1
    
    with open(file_dir, 'r') as f:
        for i in range(num_files):
            with open(f'{file_dir}.{i:02d}', 'w') as f_out:
                for _ in range((count // num_files)):
                    f_out.write(f.readline())
                if i == num_files - 1:
                    line = f.readline()
                    while line:
                        f_out.write(line)
                        line = f.readline()


def clean(src_file: str, tgt_file: str, num_files: int, num_threads: int):
    with mp.Pool(num_threads) as pool:
        res = []
        for i in range(num_files):
            res.append(pool.apply_async(clean_worker, args=(src_file, tgt_file, i)))
        for r in res:
            r.get()


def merge(file_dir: str, cleaned_file_dir: str, num_files: int):
    with open(cleaned_file_dir, 'w') as f:
        for i in range(num_files):
            cleaned_subfile = f'{file_dir}.{i:02d}.cleaned'
            subfile = f'{file_dir}.{i:02d}'
            with open(cleaned_subfile, 'r') as f_sub:
                f.write(f_sub.read() + ('\n' if i < num_files - 1 else ''))
            os.remove(cleaned_subfile)
            os.remove(subfile)


def get_bpe_tokenizer(cfg: Config) -> Tuple[Tokenizer, Trainer]:
    tokenizer = tokenizers.Tokenizer(BPE(unk_token=SPECIAL_TOKENS.UNK))
    trainer = BpeTrainer(
        special_tokens=SPECIAL_TOKENS.ALL,
        vocab_size=cfg.data.tokenizer.vocab_size,
        min_frequency=cfg.data.tokenizer.min_freq,
        show_progress=True
    )
    tokenizer.normalizer = tokenizers.normalizers.Sequence([tokenizers.normalizers.NFKC()])
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Metaspace()
    tokenizer.decoder = tokenizers.decoders.Metaspace()
    return tokenizer, trainer


def get_wordpiece_tokenizer(cfg: Config) -> Tuple[Tokenizer, Trainer]:
    tokenizer = tokenizers.Tokenizer(WordPiece(unk_token=SPECIAL_TOKENS.UNK))
    trainer = WordPieceTrainer(
        vocab_size=cfg.data.tokenizer.vocab_size,
        special_tokens=SPECIAL_TOKENS.ALL,
        min_frequency=cfg.data.tokenizer.min_freq,
        show_progress=True
    )
    tokenizer.normalizer = tokenizers.normalizers.Sequence([tokenizers.normalizers.NFKC()])
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    tokenizer.decoder = tokenizers.decoders.WordPiece(cleanup=False)
    return tokenizer, trainer


def convert(files, tokenizer, truncate, out_dir):
    def write(data, f_out):
        for i in range(len(data)):
            f_out.write(struct.pack('>i', len(data[i])))
            f_out.write(struct.pack(f'{len(data[i])}H', *data[i]))

    token_sos_id = tokenizer.token_to_id(SPECIAL_TOKENS.SOS)
    token_eos_id = tokenizer.token_to_id(SPECIAL_TOKENS.EOS)
    token_counts = []
    
    for file in files:
        bin_dir = os.path.join(out_dir, os.path.basename(file.rstrip("/")))
        with open(file, "r", encoding="utf-8") as f, open(bin_dir, "wb") as f_out:
            lines = []
            line = f.readline()
            while line:
                lines.append(line.rstrip("\n"))
                if len(lines) > 10000:
                    encodings = tokenizer.encode_batch(lines)
                    write([([token_sos_id] + list(x.ids)[:truncate] + [token_eos_id]) for x in encodings], f_out)
                    token_counts.extend([(len(x.ids) + 2) for x in encodings])
                    lines = []
                line = f.readline()
            if len(lines):
                encodings = tokenizer.encode_batch(lines)
                write([([token_sos_id] + list(x.ids)[:truncate] + [token_eos_id]) for x in encodings], f_out)
                token_counts.extend([(len(x.ids) + 2) for x in encodings])
    
    logger.info(f"Processed files: {files}")
    logger.info(f"Token count: {sum(token_counts)}")
    logger.info(f"Average token count: {sum(token_counts) / len(token_counts):.2f}")
    logger.info(f"Percentage of tokens truncated: {sum(1 for x in token_counts if x > truncate) / len(token_counts) * 100:.2f}%")


def main():
    cfg = parse_config(load_train_cfg=False, load_eval_cfg=False)
    
    logger.info(cfg)
    logger.info(f'Data directory: {cfg.data.data_dir}')
    logger.info(f'Source language: {cfg.data.src_lang}')
    logger.info(f'Target language: {cfg.data.tgt_lang}')

    num_threads = min(max(mp.cpu_count(), 1), 8)
    num_files = num_threads * 4
    logger.info(f'Number of threads: {num_threads}')
    
    logger.info('------ Clean data ------')
    cleaned_files = {}
    start = time.time()
    for split in ['train', 'valid', 'test']:
        src_file = os.path.join(cfg.data.data_dir, 'tmp', f'{split}.{cfg.data.src_lang}')
        tgt_file = os.path.join(cfg.data.data_dir, 'tmp', f'{split}.{cfg.data.tgt_lang}')
        cleaned_src_file = os.path.join(cfg.data.data_dir, f'{split}.{cfg.data.src_lang}')
        cleaned_tgt_file = os.path.join(cfg.data.data_dir, f'{split}.{cfg.data.tgt_lang}')
        cleaned_files[split] = [cleaned_src_file, cleaned_tgt_file]
        if os.path.exists(cleaned_src_file) and os.path.exists(cleaned_tgt_file):
            logger.info(f'Skipping {split} data cleaning as cleaned files already exist')
            continue
        split_to_multi_files(src_file, num_files=num_files)
        split_to_multi_files(tgt_file, num_files=num_files)
        clean(src_file, tgt_file, num_files, num_threads)
        merge(src_file, cleaned_src_file, num_files)
        merge(tgt_file, cleaned_tgt_file, num_files)
        logger.info(f'Cleaned {split} data saved to {cleaned_src_file} and {cleaned_tgt_file}')
    logger.info(f'------ Data cleaning done in {time.time() - start:.2f} s ------')

    logger.info('------ Train tokenizer ------')
    start = time.time()
    os.makedirs(cfg.data.output_dir, exist_ok=True)
    tokenizer_dir = os.path.join(cfg.data.output_dir, 'tokenizer')
    if os.path.exists(tokenizer_dir):
        logger.info('Skipping tokenizer training as tokenizer already exists')
        tokenizer = Tokenizer.from_file(tokenizer_dir)
    else:
        match cfg.data.tokenizer.model:
            case 'bpe':
                tokenizer, trainer = get_bpe_tokenizer(cfg)
            case 'wordpiece':
                tokenizer, trainer = get_wordpiece_tokenizer(cfg)
            case _:
                raise ValueError(f'Invalid tokenizer model: {cfg.data.tokenizer.model}')
        tokenizer.train(files=cleaned_files['train'] + cleaned_files['valid'], trainer=trainer)
        tokenizer.save(tokenizer_dir)
        logger.info(f'Trained tokenizer saved to {tokenizer_dir}')
    logger.info(f'------ Tokenizer training done in {time.time() - start:.2f} s ------')

    logger.info('------ Convert data to binary format ------')
    start = time.time()
    for split in ['train', 'valid', 'test']:
        if os.path.exists(os.path.join(cfg.data.output_dir, f'{split}.{cfg.data.src_lang}')):
            logger.info(f'Skipping {split} data conversion as converted files already exist')
            continue
        convert(cleaned_files[split], tokenizer, cfg.data.truncate, cfg.data.output_dir)
    logger.info(f'------ Data conversion done in {time.time() - start:.2f} s ------')

    with open(os.path.join(cfg.data.output_dir, 'config.toml'), 'wb') as f:
        tomli_w.dump(cfg.data.model_dump(), f)


if __name__ == '__main__':
    main()
