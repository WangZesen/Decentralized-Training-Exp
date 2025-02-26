import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import sys
import numpy as np
import torch
from tqdm import tqdm
from src.conf import Config, SPECIAL_TOKENS, parse_config
import pandas as pd
from src.model import get_model
from src.data import get_dataset, get_dataloader
from src.utils import batch_beam_search
from tokenizers import Tokenizer
import evaluate
from loguru import logger
logger.remove()
logger.add(sys.stdout)

'''
    Helper functions
'''

def convert_state_dict(state_dict: dict) -> dict:
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module.") or k.startswith("_model."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def read_train_log(log_file_dir: str):
    data = pd.read_csv(log_file_dir, sep=',')
    logger.info(f'Read training log from {log_file_dir}')
    logger.info(f'\tNumber of epochs: {len(data)} (from {data["epoch"].min()} to {data["epoch"].max()})')
    logger.info(f'\tNumber of steps: {data["step"].max()}')
    return data


def read_test_log(log_file_dir: str):
    data = pd.read_csv(log_file_dir, sep=',')
    logger.info(f'Read test log from {log_file_dir}')
    logger.info(f'\tNumber of epochs: {len(data)} (from {data["epoch"].min()} to {data["epoch"].max()})')
    logger.info(f'\tNumber of steps: {data["step"].max()}')
    return data


def main():
    '''
        Load configuration
    '''
    cfg = parse_config()
    cfg.train.max_tokens_per_local_batch = 512

    '''
        Read the training and test logs
    '''

    train_log_dir = os.path.join(cfg.eval.exp_dir, "train_log.csv")
    assert os.path.exists(train_log_dir), f"Train log file {train_log_dir} does not exist."
    train_log = read_train_log(train_log_dir)

    test_log_dir = os.path.join(cfg.eval.exp_dir, "test_log.csv")
    if os.path.exists(test_log_dir):
        test_log = read_test_log(test_log_dir)
    else:
        columns = train_log.columns.to_list()
        columns.append('BLEU Score')
        test_log = pd.DataFrame(columns=columns)
    out_log = pd.DataFrame(columns=test_log.columns)
    cnt = 0

    '''
        Load test dataset
    '''
    tokenizer_dir = cfg.data.output_dir
    tokenizer = Tokenizer.from_file(os.path.join(tokenizer_dir, 'tokenizer'))
    token_sos_id = tokenizer.token_to_id(SPECIAL_TOKENS.SOS)
    token_eos_id = tokenizer.token_to_id(SPECIAL_TOKENS.EOS)
    token_pad_id = tokenizer.token_to_id(SPECIAL_TOKENS.PAD)

    test_dataset = get_dataset(cfg, tokenizer_dir, split="test")
    test_ds = get_dataloader(cfg, test_dataset, tokenizer, SPECIAL_TOKENS.PAD, drop_last=False)

    '''
        Initialize Model
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(cfg, tokenizer, SPECIAL_TOKENS.PAD)
    model = model.to(device)

    # Make sure the results are reproducible
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

    '''
        Load Metric
    '''
    bleu = evaluate.load("bleu")

    '''
        Evaluate checkpoints
    '''

    for i in range(len(train_log)):
        if train_log.iloc[i]['epoch'] in test_log['epoch'].values:
            logger.info(f'Skip epoch {train_log.iloc[i]["epoch"]} as it has been evaluated.')
            out_log.loc[cnt] = test_log[test_log['epoch'] == train_log.iloc[i]['epoch']].values.tolist()[0]
            cnt += 1
            continue

        checkpoint_dir = train_log.iloc[i]['checkpoint_dir']
        if (len(checkpoint_dir) == 0) or (not os.path.exists(checkpoint_dir)) or (not os.path.isfile(checkpoint_dir)):
            logger.info(f'Skip epoch {train_log.iloc[i]["epoch"]} as the checkpoint does not exist.')
            out_log.loc[cnt] = train_log.iloc[i].values.tolist() + [pd.NA]
            cnt += 1
            continue

        state_dict = torch.load(checkpoint_dir, map_location=device)['model_state_dict']
        model.load_state_dict(convert_state_dict(state_dict))
        model.eval()

        predictions = []
        references = []

        for (src, tgt, n_tokens) in tqdm(test_ds):
            src = src.to(device)
            tgt = tgt.to(device)

            seq_lens, pred = batch_beam_search(
                model,
                src,
                token_sos_id,
                token_eos_id,
                token_pad_id,
                beam_size=cfg.eval.beam_size,
                len_penalty=cfg.eval.length_penalty,
                tolerance=cfg.eval.tolerance
            )

            src = src.numpy(force=True)
            seq_lens = seq_lens.numpy(force=True).astype(np.int32)
            pred = pred.numpy(force=True)
            tgt = tgt.numpy(force=True)
            
            for j in range(src.shape[0]):
                idx = j * cfg.eval.beam_size
                predictions.append(
                    tokenizer.decode(pred[idx, :seq_lens[idx, 0]])
                )
                
                references.append([
                    tokenizer.decode(tgt[j, :])
                ])

        result = bleu.compute(predictions=predictions, references=references)
        values = train_log.iloc[i].values.tolist()
        values.append(result['bleu'])

        out_log.loc[cnt] = values
        cnt += 1

        print(out_log)
        out_log.to_csv(os.path.join(cfg.eval.exp_dir, 'test_log.csv'), sep=',', index=False)

        os.makedirs(os.path.join(cfg.eval.exp_dir, 'inference'), exist_ok=True)
        with open(os.path.join(cfg.eval.exp_dir, 'inference', f'{train_log.iloc[i]["epoch"]}.ref.txt'), 'w') as f:
            for ref in references:
                f.write(ref[0] + '\n')
        with open(os.path.join(cfg.eval.exp_dir, 'inference', f'{train_log.iloc[i]["epoch"]}.hyp.txt'), 'w') as f:
            for hyp in predictions:
                f.write(hyp + '\n')


if __name__ == '__main__':
    main()
