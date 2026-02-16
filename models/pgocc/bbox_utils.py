import torch


def encode_3dgs(positions, scales, pc_range=None):
    xyz = positions.clone()
    if scales is not None:
        max_scales, _ = torch.max(scales, dim=2, keepdim=True)
        scales = max_scales.repeat(1, 1, 3)
    else:
        scales = torch.ones_like(xyz, device=xyz.device) * 1.6

    scales = scales.log()
    if pc_range is not None:
        xyz[..., 0] = (xyz[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
        xyz[..., 1] = (xyz[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1])
        xyz[..., 2] = (xyz[..., 2] - pc_range[2]) / (pc_range[5] - pc_range[2])

    return torch.cat([xyz, scales], dim=-1)


def encode_bbox(bboxes, pc_range=None):
    xyz = bboxes[..., 0:3].clone()
    wlh = bboxes[..., 3:6].log()
    rot = bboxes[..., 6:10]

    if pc_range is not None:
        xyz[..., 0] = (xyz[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
        xyz[..., 1] = (xyz[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1])
        xyz[..., 2] = (xyz[..., 2] - pc_range[2]) / (pc_range[5] - pc_range[2])

    return torch.cat([xyz, wlh, rot], dim=-1)


def decode_bbox(bboxes, pc_range=None):
    xyz = bboxes[..., 0:3].clone()
    wlh = bboxes[..., 3:6].exp()
    rot = bboxes[..., 6:10]

    if pc_range is not None:
        xyz[..., 0] = xyz[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
        xyz[..., 1] = xyz[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
        xyz[..., 2] = xyz[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]

    return torch.cat([xyz, wlh, rot], dim=-1)


def bbox2occrange(bboxes, occ_size, query_cube_size=None):
    xyz = bboxes[..., 0:3].clone()
    if query_cube_size is not None:
        wlh = torch.zeros_like(xyz)
        wlh[..., 0] = query_cube_size[0]
        wlh[..., 1] = query_cube_size[1]
        wlh[..., 2] = query_cube_size[2]
    else:
        wlh = bboxes[..., 3:6]
        wlh[..., 0] = wlh[..., 0] * occ_size[0]
        wlh[..., 1] = wlh[..., 1] * occ_size[1]
        wlh[..., 2] = wlh[..., 2] * occ_size[2]

    xyz[..., 0] = xyz[..., 0] * occ_size[0]
    xyz[..., 1] = xyz[..., 1] * occ_size[1]
    xyz[..., 2] = xyz[..., 2] * occ_size[2]

    xyz = torch.round(xyz)

    low_bound = torch.round(xyz - wlh / 2)
    high_bound = torch.round(xyz + wlh / 2)

    return torch.cat((low_bound, high_bound), dim=-1).long()


def occrange2bbox(occ_range, occ_size, pc_range):
    xyz = (occ_range[..., :3] + occ_range[..., 3:]).to(torch.float32) / 2
    xyz[..., 0] /= occ_size[0]
    xyz[..., 1] /= occ_size[1]
    xyz[..., 2] /= occ_size[2]
    wlh = (occ_range[..., 3:] - occ_range[..., :3]).to(torch.float32)
    wlh[..., 0] *= (pc_range[3] - pc_range[0]) / occ_size[0]
    wlh[..., 1] *= (pc_range[4] - pc_range[1]) / occ_size[1]
    wlh[..., 2] *= (pc_range[5] - pc_range[2]) / occ_size[2]
    return torch.cat((xyz, wlh), dim=-1)
