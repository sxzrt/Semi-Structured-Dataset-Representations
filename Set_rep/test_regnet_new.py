from __future__ import print_function

import sys

sys.path.append(".")
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from regression_new import RegNet_new
from regression_new import Set_Rpresentation
import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def getGreedyPerm(D):
    """
    A Naive O(N^2) algorithm to do furthest points sampling

    Parameters
    ----------
    D : ndarray (N, N)
        An NxN distance matrix for points
    Return
    ------
    tuple (list, list)
        (permutation (N-length array of indices),
        lambdas (N-length array of insertion radii))
    """

    N = D.shape[0]
    # By default, takes the first point in the list to be the
    # first point in the permutation, but could be random
    perm = np.zeros(N, dtype=np.int64)
    lambdas = np.zeros(N)
    ds = D[random.randint(0,N), :]
    for i in range(1, N):
        idx = np.argmax(ds)
        perm[i] = idx
        lambdas[i] = ds[idx]
        ds = np.minimum(ds, D[idx, :])
    return (perm, lambdas)

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def cal_pairwise_dist(x):
    """
    caculate pairwise distance
    :param x: matrix
    :return: distance
    """
    sum_x = np.sum(np.square(x), 1)
    dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
    return dist


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Regression')
    parser.add_argument('--batch-size', type=int, default=25, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=60, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    act = np.load('./FD/cifar_test_fea.npy')
    mean_t = np.mean(act, axis=0)
    var_t = np.cov(act, rowvar=False)
    m1 = np.load('./FD/dataset_feature2/train_mean.npy')
    s1 = np.load('./FD/dataset_feature2/train_variance.npy')
    act1 = np.load('./FD/dataset_feature2/train_feature.npy')

    std1 = np.std(act1.T,1)

    his_shape = []
    norm_fea = (act - m1) / std1
    for fea in norm_fea.T:
        n, bins, patches =plt.hist(fea, 30, density=True)
        value = n * np.diff(bins)
        his_shape.append(value)

    number_cluster=10
    dis = cal_pairwise_dist(act)
    (perm, lambdas) = getGreedyPerm(dis)
    estimator = KMeans(n_clusters=number_cluster)
    estimator.fit(act)
    label_pred = estimator.labels_
    act2_cluster = []
    for k in range(number_cluster):
        # initializatn of the first seed cluster 0
        initial_feature = np.mean(act[(label_pred == k)], 0)
        act2_cluster.append(initial_feature)

    fps = act[perm[0:100], :]



    model = RegNet_new().to(device)
    gap=0
    if gap==0:
        model.load_state_dict(torch.load('./FD/set_representation_model/cifar_regnet_480_2.pt', map_location=torch.device('cpu')))
        mean_i = mean_t
        var_i = var_t

    model.eval()

    with torch.no_grad():
        # testing on real data
        print('CIFAR-10.1')
        pre=model(torch.as_tensor(np.array(his_shape).T, dtype=torch.float).view(1, 1, 30, 64).cuda(),
              torch.as_tensor(np.array(act2_cluster), dtype=torch.float).view(1, 1, 10, 64).cuda(),
              torch.as_tensor(fps, dtype=torch.float).view(1, 1, 100, 64).cuda()
              )
        RMSE = mean_squared_error(np.array([87.65]), pre.cpu().numpy(), squared=False)
        print(pre)
        print(RMSE)
if __name__ == '__main__':
    main()
