############################################################
# In this script we show how to get independent mutations. #
# The implementation is based on                           #
# https://github.com/zhujiapeng/LowRankGAN                 #
# and https://github.com/zhujiapeng/resefa                 #
############################################################

import os
from tqdm import tqdm
import numpy as np

import torch
from torch.autograd.functional import jacobian

from LRF import RobustPCA

def batched_jacobian(f, x):
    """Computes the Jacobian of f w.r.t x.

    This is according to the reverse mode autodiff rule,

    sum_i v^b_i dy^b_i / dx^b_j = sum_i x^b_j R_ji v^b_i,

    where:
    - b is the batch index from 0 to B - 1
    - i, j are the vector indices from 0 to N-1
    - v^b_i is a "test vector", which is set to 1 column-wise to obtain the correct
        column vectors out ot the above expression.

    :param f: function R^N -> R^N
    :param x: torch.tensor of shape [B, N]
    :return: Jacobian matrix (torch.tensor) of shape [B, N, N]
    """
    x.requires_grad = True
    B, N = x.shape
    y = f(x)
    jacobian = list()
    for i in range(N):
        v = torch.zeros_like(y)
        v[:, i] = 1.
        dy_i_dx = torch.autograd.grad(
                    y,
                    x,
                    grad_outputs=v,
                    retain_graph=True,
                    create_graph=True,
                    allow_unused=True
                )[0]  # shape [B, N]
        jacobian.append(dy_i_dx)

    jacobian = torch.stack(jacobian, dim=2).requires_grad_()

    return jacobian

def Jacobian(G, latent_zs):
    jacobians = []
    for idx in tqdm(range(latent_zs.shape[0])):
        latent_z = latent_zs[idx:idx+1]
        jac_i = jacobian(
                    func=G,
                    inputs=latent_z,
                    create_graph=False,
                    strict=False
                )
        # print('jac_i: ', jac_i.size())
        jacobians.append(jac_i)
    jacobians = torch.cat(jacobians, dim=0)
    print('jacobians size: ', jacobians.size())
    np_jacobians = jacobians.detach().cpu().numpy()
    return np_jacobians

def Jacobian_Y(G, latent_zs, ys):
    jacobians = []
    for idx in tqdm(range(latent_zs.shape[0])):
        latent_z = latent_zs[idx:idx+1]
        y = ys[idx:idx+1]
        jac_i = jacobian(
                    func=G,
                    inputs=(latent_z, y),
                    create_graph=False,
                    strict=False
                )
        # print('jac_i: ', jac_i.size())
        jacobians.append(jac_i[0])
    jacobians = torch.cat(jacobians, dim=0)
    print('jacobians size: ', jacobians.size())
    np_jacobians = jacobians.detach().cpu().numpy()
    return np_jacobians

def get_direction(jacobians, save_dir,
        foreground_ind=None,
        background_ind=None,
        lamb=60,
        num_relax=0,
        max_iter=10000):
    # lamb: the coefficient to control the sparsity
    # num_relax: factor of relaxation for the non-zeros singular values
    image_size = jacobians.shape[2]
    z_dim = jacobians.shape[-1]
    for ind in tqdm(range(jacobians.shape[0])):
        jacobian = jacobians[ind]
        if foreground_ind is not None and background_ind is not None:
            if len(jacobian.shape) == 4:  # [H, W, 1, latent_dim]
                jaco_fore = jacobian[foreground_ind[0], foreground_ind[1], 0]
                jaco_back = jacobian[background_ind[0], background_ind[1], 0]
            elif len(jacobian.shape) == 5:  # [channel, H, W, 1, latent_dim]
                jaco_fore = jacobian[:, foreground_ind[0], foreground_ind[1], 0]
                jaco_back = jacobian[:, background_ind[0], background_ind[1], 0]
            else:
                raise ValueError(f'Shape of Jacobian is not correct!')
            jaco_fore = np.reshape(jaco_fore, [-1, z_dim])
            jaco_back = np.reshape(jaco_back, [-1, z_dim])
            coef_f = 1 / jaco_fore.shape[0]
            coef_b = 1 / jaco_back.shape[0]
            M_fore = coef_f * jaco_fore.T.dot(jaco_fore)
            B_back = coef_b * jaco_back.T.dot(jaco_back)
            # R-PCA on foreground
            RPCA = RobustPCA(M_fore, lamb=1/lamb)
            L_f, _ = RPCA.fit(max_iter=max_iter)
            rank_f = np.linalg.matrix_rank(L_f)
            # R-PCA on background
            RPCA = RobustPCA(B_back, lamb=1/lamb)
            L_b, _ = RPCA.fit(max_iter=max_iter)
            rank_b = np.linalg.matrix_rank(L_b)
            # SVD on the low-rank matrix
            _, _, VHf = np.linalg.svd(L_f)
            _, _, VHb = np.linalg.svd(L_b)
            F_principal = VHf[:rank_f]
            relax_subspace = min(max(1, rank_b - num_relax), z_dim-1)
            B_null = VHb[rank_b:].T

            F_principal_proj = B_null.dot(B_null.T).dot(F_principal.T)  # Projection
            F_principal_proj = F_principal_proj.T
            F_principal_proj /= np.linalg.norm(
                F_principal_proj, axis=1, keepdims=True)
            print('direction size: ', F_principal_proj.shape)
            if save_dir is not None:
                save_name = '%d_direction.npy' % ind
                np.save(save_dir + save_name, F_principal_proj)
            return F_principal_proj
        else:
            jaco = np.reshape(jacobian, [-1, z_dim])
            coef = 1 / jaco.shape[0]
            M = coef * jaco.T.dot(jaco)

            RPCA = RobustPCA(M, lamb=1/lamb)
            L, _ = RPCA.fit(max_iter=max_iter)
            rank = np.linalg.matrix_rank(L)
            _, _, VH = np.linalg.svd(L)
            principal = VH[:max(rank, 5)]
            print('direction size: ', principal.shape)
            if save_dir is not None:
                save_name = '%d_direction.npy' % ind
                np.save(save_dir + save_name, principal)
            return principal

if __name__ == '__main__':
    G = None # The generative model
    z = None # The latent point
    y = None # The class label

    ########################################################
    # 1. You need to first compute the Jacobian matrix.    #
    ########################################################

    # 1.1 For conventional generative model G
    J = Jacobian(G, z)

    # 1.2 For class-conditional G (e.g., BigGAN), which generates images of class `y`
    J = Jacobian(G, z, y)

    #############################################################
    # 2.1 Then, you can get the mutating directions as follows: #
    #############################################################
    directions = get_direction(J, save_dir=None)

    ##############################################################################
    # 2.2 For local mutations, you need to manually set the indexes of foreground #
    # and background. LowRankGAN authors provide examples of the indexes, see     #
    # https://github.com/zhujiapeng/resefa/blob/main/coordinate.py and            #
    # https://github.com/zhujiapeng/LowRankGAN/blob/master/coordinate.py          #
    ###############################################################################

    COORDINATE_ffhq = {
        'left_eye': [120, 95, 20, 38],
        'right_eye': [120, 159, 20, 38],
        'eyes': [120, 128, 20, 115],
        'nose': [142, 131, 40, 46],
        'mouth': [184, 127, 30, 70],
        'chin': [217, 130, 42, 110],
        'eyebrow': [126, 105, 15, 118],
    }

    def get_mask_by_coordinates(image_size, coordinate):
        """Get mask using the provided coordinates."""
        mask = np.zeros([image_size, image_size], dtype=np.float32)
        center_x, center_y = coordinate[0], coordinate[1]
        crop_x, crop_y = coordinate[2], coordinate[3]
        xx = center_x - crop_x // 2
        yy = center_y - crop_y // 2
        mask[xx:xx + crop_x, yy:yy + crop_y] = 1.
        return mask

    coords = COORDINATE_ffhq['eyes']
    mask = get_mask_by_coordinates(256, coordinate=coords)
    foreground_ind = np.where(mask == 1)
    background_ind = np.where((1 - mask) == 1)
    directions = get_direction(J, None, foreground_ind, background_ind)

    ##################################################################################
    # 3. Once you get the direction, you can perform mutations in the following way. #  
    ##################################################################################

    delta = 1.0
    for i in range(len(directions)):
        v = directions[i]
        x_ = G(z + delta * v)
        # `x_` is the mutated input using the i-th mutating direction

