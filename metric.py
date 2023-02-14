import torch


def mse_to_psnr(mse):
    psnr = - 10 * torch.log(mse) / torch.log(torch.Tensor([10.]))
    return psnr