import os

import numpy as np
from scipy import linalg
from tqdm import trange
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans

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

if __name__ == '__main__':
    test_dirs = sorted(os.listdir('dataset_feature/'))
    #'../dataset_bg'))
    fid_bg = []
    m1 = np.load('train_mean.npy')
    s1 = np.load('train_variance.npy')
    act1 = np.load('train_feature.npy')
    std1 = np.std(act1.T,1)
    feat_path = './feature/set_represeantaion/'
    number_cluster=10
    for i in range(len(test_dirs)):
        # path = test_dirs[i]
        path = i
        m2 = np.load('./feature/set_represeantaion/_%s_mean.npy' % (path))
        s2 = np.load('./feature/set_represeantaion/_%s_variance.npy' % (path))
        act2 = np.load('./feature/set_represeantaion/_%s_feature.npy' % (path))

        his_shape=[]
        norm_fea = (act2 - m1) / std1
        for fea in norm_fea.T:
            n, bins, patches = plt.hist(fea, 30, density=True)
            value= n * np.diff(bins)
            his_shape.append(value)

        dis=cal_pairwise_dist(act2)
        (perm, lambdas) = getGreedyPerm(dis)
        estimator = KMeans(n_clusters=number_cluster)
        estimator.fit(act2)
        label_pred = estimator.labels_
        act2_cluster = []
        for k in range(number_cluster):
            # initializatn of the first seed cluster 0
            initial_feature = np.mean(act2[(label_pred == k)],0)
            act2_cluster.append(initial_feature)


        fps=act2[perm[0:100],:]  # 100 for cifar

        np.save(feat_path + '_%s_shape' % (i), np.array(his_shape).T)
        np.save(feat_path + '_%s_cluster' % (i), np.array(act2_cluster))
        np.save(feat_path + '_%s_fps' % (i), fps)


