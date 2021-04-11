# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from copy import deepcopy
from pathlib import Path
import torch
from torch import nn


def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape
    each (keypoints - image_center) / max(height, width)*0.7
    I don't why do this'
    """
    _, _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one*width, one*height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0) # make last mlp's bias to zero

    def forward(self, kpts, scores):
        # scores = torch.rand(scores.size()).to(scores.device)
        # scores = torch.zeros(scores.size()).to(scores.device)
        # scores = torch.ones(scores.size()).to(scores.device)
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)] # it will be [1, 3, N]
        return self.encoder(torch.cat(inputs, dim=1))


def attention(query, key, value):
    """
    
    Args:
        query: [1, 64, 4, N]
        key: [1, 64, 4, M]
        value: [1, 64, 4, M]

    Returns: [1, 64, 4, N], [1, 4, N, M]

    """
    dim = query.shape[1] # dim: 64
    # scores - [1, 4, N, M] is query scores
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5 # dim**.5 is 8;
    # prob - [1, 4, N, M]; prob[0, 0, 0, :].sum() is 1
    prob = torch.nn.functional.softmax(scores, dim=-1)
    # weight sum value and the prob is useless afterward
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0 # 256 % 4 == 0
        self.dim = d_model // num_heads # self.dim: 64
        self.num_heads = num_heads # self.num_heads: 4
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1) # mlp (256, 256)
        # repeat merge 3 times
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        """
        merge feature for x and value
        Args:
            query: [1, 256, N]
            key: [1, 256, M]
            value: [1, 256, M]

        Returns: [1, 256, N]

        """
        batch_dim = query.size(0)
        # Do mlp for [1, 256, N] feature to extract another [1, 256, N] feature
        #   then view to [1, 64, 4, N]
        # There have 3 mlp in proj, this is assign each (query, key, value) one mlp,
        #   and conv them progressively; l is a 256-256 mlp
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        # x - [1, 64, 4, N]
        # prob - [1, 4, N, M] which is useless
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim]) # [512-512-256]
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source) # message - [1, 256, N]
        # Do mlp again to get final return [1, 256, N]
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4) # multi-head attention with 4 heads
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0, desc1):
        """
        extract descriptor crazy
        Args:
            desc0: [1, 256, N]
            desc1: [1, 256, M]

        Returns: [1, 256, N], [1, 256, M]

        """
        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            # delta0 - [1, 256, N/M]
            # delta1 - [1, 256, N/M]
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """
    Perform Sinkhorn Normalization in Log-space for stability
    Args:
        Z: [1, 240, 245]
        log_mu: [1, 240]
        log_nu: [1, 245]
        iters: int

    Returns: [1, 240, 245]

    """
    # u - [1, 240] v - [1, 245]
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2) # u - [1, 240]
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1) # v - [1, 245]
    return Z + u.unsqueeze(2) + v.unsqueeze(1) # return [1, 240, 245]


def log_optimal_transport(scores, alpha, iters: int):
    """
    Perform Differentiable Optimal Transport in Log-space for stability
    Args:
        scores: [1, m, n]
        alpha: called bin_score in another place which is z used to fill dustbins
        iters:

    Returns:

    """
    b, m, n = scores.shape # b: 1 m: 239 n: 244
    one = scores.new_tensor(1) # one: tensor(1., device='cuda:0')
    # ms: tensor(239., device='cuda:0') ns: tensor(244., device='cuda:0')
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    # couplings = scores + dustbins
    # couplings - [1, m+1, n+1]
    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log() # norm: tensor(-6.1800, device='cuda:0')
    # log_mu - [m+1]; all value is 'norm' expect last one is ns.log+norm
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    # log_nu - [n+1]; all value is 'norm' expect last one is ms.log+norm
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    # log_mu - [1, m+1] log_nu - [1, n+1]
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters) # Z - [1, m+1, n+1]
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


class SuperGlue(nn.Module):
    """SuperGlue feature matching middle-end

    Given two sets of keypoints and locations, we determine the
    correspondences by:
      1. Keypoint Encoding (normalization + visual feature and location fusion)
      2. Graph Neural Network with multiple self and cross-attention layers
      3. Final projection layer
      4. Optimal Transport Layer (a differentiable Hungarian matching algorithm)
      5. Thresholding matrix based on mutual exclusivity and a match_threshold

    The correspondence ids use -1 to indicate non-matching points.

    Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. SuperGlue: Learning Feature Matching with Graph Neural
    Networks. In CVPR, 2020. https://arxiv.org/abs/1911.11763

    """
    default_config = {
        'descriptor_dim': 256,
        'weights': 'indoor',
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.kenc = KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])

        self.gnn = AttentionalGNN(
            self.config['descriptor_dim'], self.config['GNN_layers'])

        self.final_proj = nn.Conv1d(
            self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)

        # bin_score - z used to fill dustbins
        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)
        # The two line code above is same is the code below
        # self.bin_score = torch.nn.Parameter(torch.tensor(1.))

        assert self.config['weights'] in ['indoor', 'outdoor']
        path = Path(__file__).parent
        path = path / 'weights/superglue_{}.pth'.format(self.config['weights'])
        self.load_state_dict(torch.load(str(path)))
        print('Loaded SuperGlue model (\"{}\" weights)'.format(
            self.config['weights']))

    def forward(self, data):
        """
        Run SuperGlue on a pair of keypoints and descriptors
        Args:
            data: contain keypoints and descriptors

        Returns:
            matches0/1 - [1, N/M]
            matching_scores0/1 - [1, N/M]

        """
        desc0, desc1 = data['descriptors0'], data['descriptors1'] # desc0:[1, 256, N]
        kpts0, kpts1 = data['keypoints0'], data['keypoints1'] # kpts0:[1, N, 2]

        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                'matches0': kpts0.new_full(shape0, -1, dtype=torch.int),
                'matches1': kpts1.new_full(shape1, -1, dtype=torch.int),
                'matching_scores0': kpts0.new_zeros(shape0),
                'matching_scores1': kpts1.new_zeros(shape1),
            }

        # Keypoint normalization.
        kpts0 = normalize_keypoints(kpts0, data['image0'].shape) # [1, N, 2]
        kpts1 = normalize_keypoints(kpts1, data['image1'].shape)

        # Keypoint MLP encoder.
        desc0 = desc0 + self.kenc(kpts0, data['scores0']) # both add_arg is [1, 256, N]
        desc1 = desc1 + self.kenc(kpts1, data['scores1'])

        # Multi-layer Transformer network.
        desc0, desc1 = self.gnn(desc0, desc1)

        # Final MLP projection.
        # mdesc0 - [1, 256, N]
        # model - [1, 256, M]
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        # Compute matching descriptor distance.
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1) # scores - [1, N, M]
        # self.config['descriptor_dim']**.5 is 16.0
        scores = scores / self.config['descriptor_dim']**.5

        # Run the optimal transport.
        scores = log_optimal_transport( # scores - [1, N+1, M+1]
            scores, self.bin_score,
            iters=self.config['sinkhorn_iterations'])

        # Get the matches with score above "match_threshold".
        max0, max1 = scores[:, :-1, :-1].max(dim=2), scores[:, :-1, :-1].max(dim=1)
        # indices0 - [1, N] indices1 - [1, M]
        indices0, indices1 = max0.indices, max1.indices
        # gather could be think the indices0 will replace each index
        #   by the element in indices1 in that dim
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0) #[1, N]
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1) #[1, M]
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero) # mscores0 - [1, N]
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero) # [1, M]
        valid0 = mutual0 & (mscores0 > self.config['match_threshold']) # [1, N]
        valid1 = mutual1 & valid0.gather(1, indices1) # [1, M]
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1)) # [1, N]
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1)) # [1, M]

        return {
            'matches0': indices0, # use -1 for invalid match
            'matches1': indices1, # use -1 for invalid match
            'matching_scores0': mscores0,
            'matching_scores1': mscores1,
        }
