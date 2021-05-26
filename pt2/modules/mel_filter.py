from functools import partial

import librosa
import torch


class MelFilter(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.register_buffer('win', torch.hann_window(config.win_length))

        self.stft = partial(
            torch.stft,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
            center=False,
            pad_mode='reflect',
            onesided=True,
            return_complex=False
        )

        self.register_buffer(
            'filters',
            torch.from_numpy(librosa.filters.mel(config.sample_rate, config.n_fft,
                             config.n_mels, config.fmin, config.fmax))
        )

    def forward(self, x):
        x = self.stft(x, window=self.win)
        spec = torch.sqrt(torch.square(x).sum(dim=-1) + 1e-9)
        mels = torch.einsum('nft,mf->nmt', spec, self.filters)
        logmels = torch.log(mels + 1e-5)
        return logmels
