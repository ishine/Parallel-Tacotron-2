import einops
import torch
import torch.nn.functional


class SwishBlock(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, output_dim)
        self.linear2 = torch.nn.Linear(output_dim, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.nn.functional.silu(x)
        x = self.linear2(x)
        x = torch.nn.functional.silu(x)
        return x


class Upsampling(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.conv1 = torch.nn.Conv1d(dim, 8, 3, 1, 1)
        self.conv2 = torch.nn.Conv1d(dim, 8, 3, 1, 1)

        self.swish_block1 = SwishBlock(8+1+1, 16)
        self.swish_block2 = SwishBlock(8+1+1, 2)
        self.projection1 = torch.nn.Linear(16, 1)
        self.projection2 = torch.nn.Linear(2, 1)

    def forward(self, T, durations: torch.Tensor, features: torch.Tensor):
        raise NotImplementedError()
        features = einops.rearrange(features, 'N W C -> N C W')
        left = torch.nn.functional.silu(self.conv1(features))
        right = torch.nn.functional.silu(self.conv2(features))
        left = einops.rearrange(left, 'N C W -> N W C')
        right = einops.rearrange(right, 'N C W -> N W C')

        token_end = torch.cumsum(durations, dim=1)
        token_start = token_end - durations

        S = T - token_start
        E = token_end - T

        import pdb
        pdb.set_trace()
        xleft = torch.cat((S, E, left), dim=-1)
        xright = torch.cat((S, E, right), dim=-1)

        C = self.swish_block1(xleft)
        xright = self.swish_block2(xright)
        xright = self.projection2(xright)
        W = torch.softmax(xright, dim=1)
        xright = torch.bmm(features, W)
        xleft = torch.einsum('nk,nkp->np', W[..., 0], C)
        xleft = self.projection1(xleft)
        O = xleft + xright
        return O
