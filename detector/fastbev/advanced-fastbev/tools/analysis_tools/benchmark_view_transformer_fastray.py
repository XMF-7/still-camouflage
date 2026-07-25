# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet benchmark a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--samples', default=1000, help='samples to benchmark')
    parser.add_argument(
        '--log-interval', default=50, help='interval of logging')
    parser.add_argument(
        '--mem-only',
        action='store_true',
        help='Conduct the memory analysis only')
    parser.add_argument(
        '--no-acceleration',
        action='store_true',
        help='Omit the pre-computation acceleration')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    if not args.no_acceleration:
        cfg.model.img_view_transformer.accelerate=True
    cfg.model.train_cfg = None
    assert cfg.model.type == 'FastBEV', \
        'Please use class FastBEV for ' \
        'view transformation inference ' \
        'speed estimation instead of %s'% cfg.model.type
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = 100
    pure_inf_time = 0
    # D = model.module.img_view_transformer.D
    out_channels = model.module.img_view_transformer.out_channels
    depth_net = model.module.img_view_transformer.depth_net
    view_transformer = model.module.img_view_transformer
    # benchmark with several samples and take the average
    for i, data in enumerate(data_loader):
        if i == 0:
            with torch.no_grad():
                img_feat, _ = \
                    model.module.image_encoder(data['img_inputs'][0][0].cuda())
                B, N, C, H, W = img_feat.shape
                x = depth_net(img_feat.reshape(B * N, C, H, W))

            input = [img_feat] + [d.cuda() for d in data['img_inputs'][0][1:] + data['lidar2image']]
            bev_feat = view_transformer(input)[0]

            img, sensor2ego, ego2global, cam2imgs, post_rots, post_trans, bda, lidar2image = input
            img = img.squeeze(0)
            img = view_transformer.depth_net(img)
            n_images, n_channels, height, width = img.shape

            img = img.permute(0, 2, 3, 1).reshape(-1, n_channels)
            voxel_x = torch.zeros(
                (int(view_transformer.grid_size[0]) * int(view_transformer.grid_size[1]) * int(view_transformer.grid_size[2]), n_channels), device=img.device
            ).type_as(img)

            pre_voxel_coors_list = view_transformer.pre_voxel_coors_list
            pre_img_coors_list = view_transformer.pre_img_coors_list
            print(pre_img_coors_list.shape)

            if view_transformer.is_transpose:
                permute = [0, 3, 2, 1]
            else:
                permute = [0, 3, 1, 2]

        torch.cuda.synchronize()
        start_time = time.perf_counter()
        with torch.no_grad():
            if view_transformer.fix_empty:
                x = img[pre_img_coors_list].view(1, *view_transformer.grid_size.int().tolist(), n_channels)
            else:
                voxel_x[pre_voxel_coors_list] = img[pre_img_coors_list]
                x = voxel_x.view(1, *view_transformer.grid_size.int().tolist(), n_channels)
            N, X, Y, Z, C = x.shape
            if view_transformer.fuse_type is not None:
                if view_transformer.fuse_type == 's2c':
                    x = x.reshape(N, X, Y, Z*C).permute(permute)
                    x = view_transformer.fuse(x)
                elif view_transformer.fuse_type == 'sum':
                    x = x.sum(dim=-2).permute(permute)
                elif view_transformer.fuse_type == 'max':
                    x = x.max(dim=-2)[0].permute(permute)
                else:
                    raise NotImplemented
            x = view_transformer.downsample(x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % args.log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(f'Done image [{i + 1:<3}/ {args.samples}], '
                      f'fps: {fps:.1f} img / s')

        if (i + 1) == args.samples:
            pure_inf_time += elapsed
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(f'Overall fps: {fps:.1f} img / s')
            return fps


if __name__ == '__main__':
    repeat_times = 1
    fps_list = []
    for _ in range(repeat_times):
        fps = main()
        time.sleep(5)
        fps_list.append(fps)
    fps_list = np.array(fps_list, dtype=np.float32)
    print(f'Mean Overall fps: {fps_list.mean():.4f} +'
          f' {np.sqrt(fps_list.var()):.4f} img / s')
