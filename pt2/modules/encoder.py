import torch

# from .residual_encoder import ResidualEncoder
from .text_encoder import TextEncoder


class Encoder(torch.nn.Module):
    def __init__(self, num_tokens, num_speakers, dim):
        super().__init__()
        self.text_encoder = TextEncoder(num_tokens, dim)
        self.speaker_embed = torch.nn.Embedding(num_speakers, dim)
        # self.residual_encoder = ResidualEncoder(dim)

    def forward(self, text, text_mask, speaker):
        encoded_text = self.text_encoder(text, text_mask)
        encoded_speaker = self.speaker_embed(speaker)
        # residual = self.residual_encoder(mel)

        # x = encoded_text + encoded_speaker + residual
        x = torch.cat(
            (torch.broadcast_to(encoded_speaker[..., None], encoded_text.shape), encoded_text),
            dim=1
        )
        return x
