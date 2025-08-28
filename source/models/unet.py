"""
https://github.com/joeylitalien/noise2noise-pytorch/blob/master/src/unet.py
https://github.com/tbepler/topaz/blob/25cb2cb8de2a0f995da42a472eb86b7011c8535a/topaz/denoise.py?fbclid=IwAR0IdzFXmH1V9qzoVKm9zX5_Vg0KUzVE8iyUTbIu266mq4vAsROdt11MEbk#L851
"""
import torch
import torch.nn as nn


class ResidualConnection(nn.Module):
    def __init__(self, block: nn.Module):
        super(ResidualConnection, self).__init__()
        self.block = block

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        res_out = self.block(torch.cat([x[0], x[1]], 1))
        return res_out + x[0]
    
class InceptionBlock(nn.Module):
    def __init__(self, layers: list[nn.Module]):
        super(InceptionBlock, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([layer(x) for layer in self.layers], dim=-3)

class ResidualConcatenation(nn.Module):
    def __init__(self, layers: list[nn.Module]):
        super(ResidualConcatenation, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layers[0](x)
        if len(self.layers) != 1:
            for layer in self.layers[1:]:
                out = torch.cat([out, layer(x)], dim=-3)
        return out

class UNet(nn.Module):
    # modified U-net from noise2noise paper
    def __init__(self, input_size: int, in_channels: int = 1, out_channels: int = 1, nf: int = 48, depth: int = 4, activation: nn.Module = nn.LeakyReLU(0.1, inplace=True)):
        super(UNet, self).__init__()

        last_size = input_size
        out_pads = [last_size]
        for _ in range(depth + 1):
            last_size = last_size // 2 + last_size % 2
            out_pads.append(last_size)

        self.activation = activation

        self.inc = nn.Sequential( nn.Conv2d(in_channels, nf, 3, padding=1, stride=1)
                                , self.activation
                                , nn.Conv2d(nf, nf, 3, padding=1, stride=1)
                                , self.activation)
        
        self.enc = nn.ModuleList([nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1, stride=2)
                                               , self.activation
                                               , nn.Conv2d(nf, nf, 3, padding=1, stride=1)
                                               , self.activation
                                 ) for i in range(depth)])

        self.mid = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1, stride=2)
                                , self.activation
                                , nn.Conv2d(nf, nf, 3, padding=1, stride=1)
                                , self.activation
                                , nn.ConvTranspose2d(nf, nf, 3, padding=1, stride=2, output_padding=1 - (out_pads[-2] % 2))
                                , self.activation
                                )
        
        self.dec = nn.ModuleList([
            nn.Sequential(
                ResidualConnection(
                    nn.Sequential( nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , self.activation
                                 , nn.Conv2d(2*nf, nf, 3, padding=1)
                                 , self.activation))
                , nn.ConvTranspose2d(nf, nf, 3, padding=1, stride=2, output_padding=1 - (out_pads[-3-i] % 2))
                , self.activation
                ) for i in range(depth)
            ] + [
                nn.Sequential(
                    ResidualConnection(
                        nn.Sequential( nn.Conv2d(2*nf+in_channels, 2*nf+in_channels, 3, padding=1)
                                     , self.activation
                                     , nn.Conv2d(2*nf+in_channels, nf, 3, padding=1)
                                     , self.activation))
                    , nn.Conv2d(nf, 64, 3, padding=1)
                    , self.activation
                )])
        
        self.outc =  nn.Sequential( nn.Conv2d(64, 32, 3, padding=1)
                                  , self.activation
                                  , nn.Conv2d(32, out_channels, 3, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        steps = [self.inc(x)]
        for encoder in self.enc:
            steps.append(encoder(steps[-1]))

        steps[0] = torch.cat([steps[0], x], 1)
        x = self.mid(steps[-1])

        for decoder in self.dec:
            x = decoder((x, steps.pop(-1)))

        x = self.outc(x)
        
        return x
