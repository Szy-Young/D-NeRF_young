import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierEmbedding:
    def __init__(self,
                 n_freq=10,
                 log_sample=True,
                 input_dim=3,
                 include_input=True):
        self.embed_fns = []
        self.output_dim = 0

        # Identity mapping
        if include_input:
            self.embed_fns.append(lambda x: x)
            self.output_dim += input_dim

        # Fourier embedding
        if log_sample:
            freq_bands = 2.**torch.linspace(0., n_freq-1, steps=n_freq)
        else:
            freq_bands = torch.linspace(2.**0., 2.**(n_freq-1), steps=n_freq)
        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq : torch.sin(freq * x))
            self.embed_fns.append(lambda x, freq=freq : torch.cos(freq * x))
            self.output_dim += 2 * input_dim

    def __call__(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class NeRF(nn.Module):
    def __init__(self,
                 n_layer=8,
                 n_dim=256,
                 input_dim=3,
                 input_view_dim=3,
                 skips=[4],
                 use_viewdir=False,
                 rgb_act='sigmoid',
                 density_act='relu'):
        super().__init__()
        self.skips = skips
        self.use_viewdir = use_viewdir

        # Point -> feature
        self.point_fc = nn.ModuleList()
        self.point_fc.append(nn.Linear(input_dim, n_dim))
        for l in range(n_layer - 1):
            if l in skips:
                self.point_fc.append(nn.Linear(n_dim + input_dim, n_dim))
            else:
                self.point_fc.append(nn.Linear(n_dim, n_dim))

        if use_viewdir:
            self.view_fc = nn.ModuleList([nn.Linear(n_dim + input_view_dim, n_dim // 2)])
            self.feat_fc = nn.Linear(n_dim, n_dim)
            self.density_fc = nn.Linear(n_dim, 1)
            self.rgb_fc = nn.Linear(n_dim // 2, 3)
        else:
            self.density_fc = nn.Linear(n_dim, 1)
            self.rgb_fc = nn.Linear(n_dim, 3)

        # # Output branch for density
        # self.density_fc = nn.Linear(n_dim, 1)
        #
        # # Output branch for RGB color
        # if use_viewdir:
        #     self.view_fc = nn.ModuleList([nn.Linear(n_dim + input_view_dim, n_dim // 2)])
        #     self.feat_fc = nn.Linear(n_dim, n_dim)
        #     self.rgb_fc = nn.Linear(n_dim // 2, 3)
        # else:
        #     self.rgb_fc = nn.Linear(n_dim, 3)

        # Output activations
        act_fns = {'identity': lambda x: x,
                   'sigmoid': lambda x: torch.sigmoid(x),
                   'relu': lambda x: F.relu(x),
                   'softplus': lambda x: F.softplus(x),
                   'shifted_softplus': lambda x: F.softplus(x - 1)}
        self.rgb_act = act_fns[rgb_act]
        self.density_act = act_fns[density_act]


    def forward(self, point, view=None, density_noise_std=0.):
        """
        :param point: (N, C) torch.Tensor.
        :param view: (N, C) torch.Tensor, if provided.
        :param density_noise_std: Noise added to raw density output
        :return:
            rgb: (N, 3) torch.Tensor.
            density: (N,) torch.Tensor.
        """
        h = point
        # Point -> feature
        for l in range(len(self.point_fc)):
            h = self.point_fc[l](h)
            h = F.relu(h)
            if l in self.skips:
                h = torch.cat([point, h], 1)

        # Output branch for density
        density = self.density_fc(h)

        # Output branch for RGB color
        if self.use_viewdir:
            feat = self.feat_fc(h)
            h = torch.cat([feat, view], 1)
            for l in range(len(self.view_fc)):
                h = self.view_fc[l](h)
                h = F.relu(h)
            rgb = self.rgb_fc(h)
        else:
            rgb = self.rgb_fc(h)

        # Add noise to raw density output
        if density_noise_std > 0.:
            noise = density_noise_std * torch.randn(density.shape)
            density += noise

        # Output activations
        rgb = self.rgb_act(rgb)
        density = self.density_act(density)
        density = density.squeeze(1)

        return rgb, density


class D_NeRF(nn.Module):
    def __init__(self,
                 n_layer=8,
                 n_dim=256,
                 input_dim=3,
                 input_view_dim=3,
                 input_time_dim=1,
                 skips=[4],
                 use_viewdir=False,
                 embedding=None,
                 rgb_act='sigmoid',
                 density_act='relu'):
        super().__init__()
        self.skips = skips

        # Positional embedding for points
        self.embedding = embedding

        # Canonical NeRF
        self.canonical = NeRF(n_layer=n_layer, n_dim=n_dim,
                              input_dim=input_dim, input_view_dim=input_view_dim,
                              skips=skips, use_viewdir=use_viewdir, rgb_act=rgb_act, density_act=density_act)

        # Deformation field
        self.timepoint_fc = nn.ModuleList()
        self.timepoint_fc.append(nn.Linear(input_dim + input_time_dim, n_dim))
        for l in range(n_layer - 1):
            if l in skips:
                self.timepoint_fc.append(nn.Linear(n_dim + input_dim, n_dim))
            else:
                self.timepoint_fc.append(nn.Linear(n_dim, n_dim))
        self.flow_fc = nn.Linear(n_dim, 3)


    def query_time(self, point, time):
        h = torch.cat([point, time], 1)
        for l in range(len(self.timepoint_fc)):
            h = self.timepoint_fc[l](h)
            h = F.relu(h)
            if l in self.skips:
                h = torch.cat([point, h], 1)
        flow = self.flow_fc(h)
        return flow


    def forward(self, point, view, time, zero_canonical=True, density_noise_std=0.):
        """
        :param point: (N, C) torch.Tensor.
        :param view: (N, C) torch.Tensor.
        :param time: (N, C) torch.Tensor.
        :return:
            rgb: (N, 3) torch.Tensor.
            density: (N,) torch.Tensor.
        """
        cur_time = torch.unique(time[:, 0])
        assert len(cur_time) == 1, 'Only accept all points from the same time!'

        if cur_time[0] == 0. and zero_canonical:
            flow = torch.zeros_like(point[:, :3])
        else:
            flow = self.query_time(point, time)
            point = point[:, :3] + flow
            point = self.embedding(point)

        rgb, density = self.canonical(point, view, density_noise_std)
        return rgb, density, flow


if __name__ == '__main__':
    torch.manual_seed(0)

    point_embedding = lambda x: x
    model = D_NeRF(use_viewdir=True, embedding=point_embedding, rgb_act='identity', density_act='identity')
    print(model.flow_fc.bias)

    point = torch.randn(8, 32, 3)
    view = torch.randn(8, 32, 3)
    time = 0.4 * torch.ones(8, 32, 1)
    point, view, time = point.reshape(-1, 3), view.reshape(-1, 3), time.reshape(-1, 1)

    rgb, density, flow = model(point, view, time)