import numpy as np
import torch

def laplacian_aleatoric_uncertainty_loss(input, target, log_variance, reduction='mean'):
    '''
    References:
        MonoPair: Monocular 3D Object Detection Using Pairwise Spatial Relationships, CVPR'20
        Geometry and Uncertainty in Deep Learning for Computer Vision, University of Cambridge
    '''
    assert reduction in ['mean', 'sum']
    loss = 1.4142 * torch.exp(-0.5*log_variance) * torch.abs(input - target) + 0.5*log_variance #u = torch.exp(0.5*log_variance)
    return loss.mean() if reduction == 'mean' else loss.sum()

# def laplacian_aleatoric_uncertainty_loss_1(input, target, log_variance, reduction='mean'):
#     assert reduction in ['mean', 'sum']
#     loss = 1.4142 * torch.exp(-0.5*log_variance) * torch.abs(input - target) + 0.5*log_variance
#     d = torch.Tensor(target.size()).cuda()
#     d[:] = 1.0
#     # d[target <= 60.0] = 1.6
#     d[target <= 50.0] = 2.0
#     d[target < 10.0] = 1.0
#     loss = loss * d
#     return loss.mean() if reduction == 'mean' else loss.sum()

def gaussian_aleatoric_uncertainty_loss(input, target, log_variance, reduction='mean'):
    '''
    References:
        What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?, Neuips'17
        Geometry and Uncertainty in Deep Learning for Computer Vision, University of Cambridge
    '''
    assert reduction in ['mean', 'sum']
    loss = 0.5 * torch.exp(-log_variance) * torch.abs(input - target)**2 + 0.5 * log_variance
    return loss.mean() if reduction == 'mean' else loss.sum()




if __name__ == '__main__':
    pass
