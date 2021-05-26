"""dataloader class."""

from pathlib import Path
import soundfile as sf
import random

import torch.utils.data
import logging

def create_train_val_dataloader(data_dir: Path, batch_size, config):
    texts = open(data_dir / 'phonemes.txt', 'r').readlines()
    filenames = open(data_dir / 'filenames.txt', 'r').readlines()
    assert len(texts) == len(filenames)
    data = []

    logging.info('Load all records to memory')
    
    for text, filename in zip(texts, filenames):
        # text to token
        text = text.strip()
        filename = filename.strip()
        token = [config.graphemes.index(c) for c in text]
        # load wav
        wav_fn = data_dir / f'wavs/{filename}.wav'
        y, sample_rate = sf.read(wav_fn, dtype='int16')
        assert sample_rate == config.sample_rate
        data.append((filename, token, y))
    
    random.Random(42).shuffle(data)
    L = len(data) * 8 // 10 # 80% data for training purpose, 20% data for validation
    train_data = data[:L]
    val_data = data[L:]

    dl1 = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, drop_last=True, collate_fn=lambda x: x)
    dl2 = torch.utils.data.DataLoader(val_data, batch_size, shuffle=False, drop_last=False, collate_fn=lambda x: x)
    return dl1, dl2
