"""main training script."""
import importlib.util
import logging
from pathlib import Path

import torch
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pad_sequence

from pt2.dataloader import create_train_val_dataloader
from pt2.modules.mel_filter import MelFilter

from .model import ParallelTacotron2


def loss_fn(model, device, melfilter, inputs):
    # prepare data
    idents, tokens, wavs = zip(*inputs)
    tokens = [torch.Tensor(t).long() for t in tokens]
    wavs = [torch.from_numpy(w) for w in wavs]
    token_lengths = torch.LongTensor([t.shape[0] for t in tokens])
    tokens = pad_sequence(tokens, batch_first=True)
    B, L = tokens.shape
    token_masks = torch.arange(0, L)[None, :] >= token_lengths[:, None]
    tokens = tokens.to(device)
    token_masks = token_masks.to(device)

    # model forward

    speaker = torch.zeros(B).long().to(device)
    mel_hats = model(tokens, token_masks, speaker)
    wavs, wav_lengths = pad_sequence(wavs, batch_first=True)
    wavs = wavs.float() / (2**15)  # only work for 16bit data
    mel_gts = melfilter(wavs.to(device))

    # mse loss
    loss = torch.mean(torch.abs(mel_hats - mel_gts))
    return loss


def train(
        model,
        args,
        config,
        train_dataloader,
        val_dataloader,
        optimizer,
        lr_scheduler,
        last_training_step,
        last_epoch
):

    step = last_training_step
    epoch = last_epoch
    device = torch.device(args.device)
    melfilter = MelFilter(config).to(device)
    logging.info('Start training')
    while step < args.training_steps:
        losses = []
        model.train()
        epoch = epoch + 1
        for batch in train_dataloader:
            step = step + 1
            loss = loss_fn(model, device, melfilter, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        train_loss = sum(losses) / len(losses)
        val_losses = []
        model.eval()
        for batch in val_dataloader:
            val_loss = loss_fn(model, device, melfilter, batch)
            val_losses.append(val_loss)
        val_loss = sum(val_losses) / len(val_losses)
        logging.info(f'step {step}  train loss {train_loss:.5f}  val loss {val_loss:.5f}')
        lr_scheduler.step(val_loss)
        if epoch % args.epochs_per_checkpoint == 0:
            fn = args.checkpoint_dir / f'{args.model_prefix}-{epoch:05d}.pth'
            logging.info(f'saving checkpoint at file {fn}')
            torch.save((step, epoch, model.state_dict(), optimizer.state_dict(), lr_scheduler.state_dict()), fn)


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('-c', '--checkpoint-dir', default='ckpts', type=Path)
    parser.add_argument('-d', '--dataset-dir', default='dataset/LJSpeech-1.1', type=Path)
    parser.add_argument('-n', '--training-steps', default=1_000_000, type=int)
    parser.add_argument('-g', '--device', default='cuda', type=str)
    parser.add_argument('-p', '--model-prefix', default='pt2', type=str)
    parser.add_argument('-l', '--learning-rate', default=1e-4, type=float)
    parser.add_argument('-w', '--weight-decay', default=1e-4, type=float)
    parser.add_argument('-f', '--epochs-per-checkpoint', default=10, type=float)
    parser.add_argument('--config-file', default='pt2/config.py', type=Path)
    args = parser.parse_args()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    # setup logging
    log_file_idx = len(tuple(args.checkpoint_dir.glob('train-*.log')))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(args.checkpoint_dir / f"train-{log_file_idx}.log"), logging.StreamHandler()]
    )

    # load config
    spec = importlib.util.spec_from_file_location("pt2_config", args.config_file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = config_module.Config
    logging.info(vars(config))

    model: torch.nn.Module = ParallelTacotron2(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    last_training_step = -1
    last_epoch = -1
    logging.info(model)

    # load latest checkpoint
    ckpts = sorted(args.checkpoint_dir.glob(f'{args.model_prefix}-*.pth'))
    if len(ckpts) > 0:
        fn = ckpts[-1].resolve()
        logging.info(f'load latest checkpoint from file: {fn}')
        last_training_step, last_epoch, model_state, optimizer_state, lr_scheduler_state = torch.load(fn)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        lr_scheduler.load_state_dict(lr_scheduler_state)

    train_dataloader, val_dataloader = create_train_val_dataloader(args.dataset_dir, args.batch_size, config)
    train(model, args, config, train_dataloader, val_dataloader,
          optimizer, lr_scheduler, last_training_step, last_epoch)


if __name__ == '__main__':
    main()
