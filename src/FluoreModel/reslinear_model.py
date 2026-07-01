from torch import nn


class ResUnit(nn.Module):
    def __init__(self, insize, outsize):
        super(ResUnit, self).__init__()
        self.layer, self.skip = self._res_unit(insize, outsize)

    def _res_unit(self, insize, outsize):
        sequential = nn.Sequential(
            nn.BatchNorm1d(insize),
            nn.Linear(insize, insize // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(insize // 2, outsize),
        )
        skip = nn.Linear(insize, outsize)
        return sequential, skip

    def forward(self, x):
        x = self.layer(x) + self.skip(x)
        return x


class ResLinear(nn.Module):
    def __init__(self, insize, nlayers):
        super(ResLinear, self).__init__()
        self.res_units = nn.Sequential(
            *[ResUnit(insize // i, insize // (i + 1)) for i in range(1, nlayers + 1)]
        )

        self.reg_layer = self._make_linear_layer(
            insize // (nlayers + 1),
            [
                insize // (nlayers + 1) // 2,
                insize // (nlayers + 1) // 4,
                insize // (nlayers + 1) // 8,
                1,
            ],
        )

    def _make_linear_layer(self, in_size, out_sizes):
        sequential = nn.Sequential(
            nn.BatchNorm1d(in_size), nn.Linear(in_size, out_sizes[0])
        )
        if len(out_sizes) == 1:
            return sequential
        for i in range(len(out_sizes) - 1):
            new_layer = nn.Linear(out_sizes[i], out_sizes[i + 1])
            if i < len(out_sizes) - 1:
                sequential.append(nn.ReLU())
            sequential.append(new_layer)
        return sequential

    def forward(self, x):
        x = self.res_units(x)
        x = self.reg_layer(x)
        return x