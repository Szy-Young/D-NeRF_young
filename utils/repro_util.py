import torch
from collections import OrderedDict


def load_official_dnerf_weights(weight_path):
    weight = torch.load(weight_path)['network_fn_state_dict']
    weight_renamed = OrderedDict()

    for k, v in weight.items():
        if k.startswith('_occ.pts_linears'):
            k = k.replace('_occ.pts_linears', 'canonical.point_fc')
        elif k.startswith('_occ.views_linears'):
            k = k.replace('_occ.views_linears', 'canonical.view_fc')
        elif k.startswith('_occ.feature_linear'):
            k = k.replace('_occ.feature_linear', 'canonical.feat_fc')
        elif k.startswith('_occ.alpha_linear'):
            k = k.replace('_occ.alpha_linear', 'canonical.density_fc')
        elif k.startswith('_occ.rgb_linear'):
            k = k.replace('_occ.rgb_linear', 'canonical.rgb_fc')
        elif k.startswith('_time_out'):
            k = k.replace('_time_out', 'flow_fc')
        elif k.startswith('_time'):
            k = k.replace('_time', 'timepoint_fc')
        weight_renamed[k] = v

    return weight_renamed