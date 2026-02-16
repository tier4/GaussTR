"""Pointops: Farthest Point Sampling CUDA extension with PyTorch fallback."""

import torch

try:
    from ._C import farthest_point_sampling_cuda

    class FarthestPointSampling(torch.autograd.Function):
        @staticmethod
        def forward(ctx, xyz, offset, new_offset):
            """
            input: xyz: (n, 3), offset: (b), new_offset: (b)
            output: idx: (m)
            """
            assert xyz.is_contiguous()
            n, b, n_max = xyz.shape[0], offset.shape[0], offset[0]
            for i in range(1, b):
                n_max = max(offset[i] - offset[i - 1], n_max)
            idx = torch.cuda.IntTensor(new_offset[b - 1].item()).zero_()
            tmp = torch.cuda.FloatTensor(n).fill_(1e10)
            farthest_point_sampling_cuda(b, n_max, xyz, offset.int(), new_offset.int(), tmp, idx)
            del tmp
            return idx

    farthest_point_sampling = FarthestPointSampling.apply
    _CUDA_AVAILABLE = True

except ImportError:
    _CUDA_AVAILABLE = False

    def farthest_point_sampling(xyz, offset, new_offset):
        """Pure PyTorch fallback for farthest point sampling.

        Args:
            xyz: (n, 3) float tensor of point coordinates
            offset: (b,) int tensor of cumulative point counts per batch
            new_offset: (b,) int tensor of cumulative sample counts per batch

        Returns:
            idx: (m,) int tensor of selected point indices
        """
        b = offset.shape[0]
        all_idx = []

        for i in range(b):
            start_n = 0 if i == 0 else offset[i - 1].item()
            end_n = offset[i].item()
            start_m = 0 if i == 0 else new_offset[i - 1].item()
            end_m = new_offset[i].item()
            num_sample = end_m - start_m

            pts = xyz[start_n:end_n]  # (n_i, 3)
            n_pts = pts.shape[0]

            if num_sample >= n_pts:
                idx = torch.arange(n_pts, device=xyz.device, dtype=torch.int)
                if num_sample > n_pts:
                    extra = torch.randint(0, n_pts, (num_sample - n_pts,),
                                          device=xyz.device, dtype=torch.int)
                    idx = torch.cat([idx, extra])
                idx = idx + start_n
            else:
                # Iterative FPS
                selected = torch.zeros(num_sample, dtype=torch.long, device=xyz.device)
                dists = torch.full((n_pts,), 1e10, device=xyz.device)
                # Start from first point
                farthest = 0
                for j in range(num_sample):
                    selected[j] = farthest
                    centroid = pts[farthest].unsqueeze(0)  # (1, 3)
                    dist = torch.sum((pts - centroid) ** 2, dim=-1)  # (n_pts,)
                    dists = torch.minimum(dists, dist)
                    farthest = torch.argmax(dists).item()
                idx = (selected + start_n).int()

            all_idx.append(idx)

        return torch.cat(all_idx)


def is_cuda_available():
    return _CUDA_AVAILABLE
