"""main training script."""
import importlib.util
import logging
from pathlib import Path

import einops
import matplotlib.pyplot as plt
import torch
from torch.functional import einsum
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pad_sequence

from pt2.dataloader import create_train_val_dataloader
from pt2.modules.mel_filter import MelFilter

from .model import ParallelTacotron2


def loss_fn(model, device, melfilter, inputs, config, return_aux=False):
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

    wav_lengths = torch.LongTensor([w.shape[0] for w in wavs])
    duration_gts = (wav_lengths.float() / config.sample_rate).to(device)
    wavs = pad_sequence(wavs, batch_first=True)
    wavs = wavs.to(device).float() / (2**15)  # only work for 16bit data
    mel_gts = melfilter(wavs)
    mel_gts = einops.rearrange(mel_gts, 'N C W -> N W C')

    # model forward

    speaker = torch.zeros(B).long().to(device)
    T = torch.arange(0, mel_gts.shape[1], dtype=torch.float32) * config.hop_length / config.sample_rate
    T = T.to(device)
    mel_hats, duration_hats = model(tokens, token_masks, speaker, T)
    duration_hats = duration_hats.squeeze(-1)
    duration_hats = torch.where(token_masks, torch.zeros_like(duration_hats), duration_hats)
    duration_hats = torch.sum(duration_hats, dim=1)

    duration_loss = torch.abs(duration_hats - duration_gts).mean()

    # mse loss
    mel_lengths = wav_lengths / config.hop_length
    L = mel_gts.shape[1]
    mel_mask = (torch.arange(0, L)[None, :] < mel_lengths[:, None]).byte().to(device)
    mel_losses = [torch.abs(mel - mel_gts).mean(-1) for mel in mel_hats]
    mel_losses = [torch.sum(l * mel_mask) / torch.sum(mel_mask) for l in mel_losses]
    mel_loss = sum(mel_losses) / len(mel_losses)
    loss = mel_loss + 100 * duration_loss
    if return_aux:
        return loss, (duration_hats, duration_gts, mel_hats[-1], mel_gts)
    else:
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
            loss = loss_fn(model, device, melfilter, batch, config)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if step % 100 == 0:
                train_loss = sum(losses) / len(losses)
                print(f' step {step:06d}  train loss {train_loss:3.5f}\r', end='')

        train_loss = sum(losses) / len(losses)
        val_losses = []
        model.eval()
        for batch in val_dataloader:
            val_loss, aux = loss_fn(model, device, melfilter, batch, config, return_aux=True)
            val_losses.append(val_loss.item())
        val_loss = sum(val_losses) / len(val_losses)
        logging.info(f'epoch {epoch:05d}  step {step:07d}  train loss {train_loss:.5f}  val loss {val_loss:.5f}')

        # logging
        plt.figure(figsize=(5, 2))
        plt.plot(aux[0].data.cpu().numpy())
        plt.plot(aux[1].data.cpu().numpy())
        plt.legend(['predicted', 'groundtruth'])
        plt.savefig(args.checkpoint_dir / f'{args.model_prefix}-{epoch:05d}-duration.png')
        plt.close()

        plt.figure(figsize=(5, 4))
        plt.subplot(2, 1, 1)
        plt.imshow(aux[2][0].data.cpu().numpy().T, aspect='auto', origin='lower')
        plt.subplot(2, 1, 2)
        plt.imshow(aux[3][0].data.cpu().numpy().T, aspect='auto', origin='lower')
        plt.savefig(args.checkpoint_dir / f'{args.model_prefix}-{epoch:05d}-melspec.png')
        plt.close()

        plt.figure(figsize=(5, 4))
        plt.imshow(model.upsampler.attention.cpu().data.numpy(), aspect='auto')
        plt.colorbar()
        plt.savefig(args.checkpoint_dir / f'{args.model_prefix}-{epoch:05d}-attention.png')
        plt.close()

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
