import einops
import torch
from torch import Tensor

from .modules.decoder import Decoder
from .modules.duration_predictor import DurationPredictor
from .modules.encoder import Encoder
from .modules.upsampling import Upsampling


class ParallelTacotron2(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder = Encoder(config.num_tokens, config.num_speakers, config.encoder_dim)
        self.encoder_projection = torch.nn.Linear(config.encoder_dim*2, config.duration_predictor_dim)
        self.duration_predictor = DurationPredictor(config.duration_predictor_dim)
        self.upsampler = Upsampling(config.upsampling_dim)
        self.decoder = Decoder(config.decoder_dim, config.decoder_num_blocks, config.n_mels)

    def forward(self, text, text_mask, speaker) -> Tensor:
        x = self.encoder(text, text_mask, speaker)
        x = einops.rearrange(x, 'N C W -> N W C')
        x = self.encoder_projection(x)
        # x = einops.rearrange(x, 'N W C -> N C W')
        durations, features = self.duration_predictor(x)
        self.upsampler(512, durations, features)
        return x
