'''create dataset and dataloader'''
import logging
import os
import sys

import torch
import torch.utils.data

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dataset.util import imresize_np
except ImportError:
    pass


def create_dataloader(dataset, dataset_opt, opt, sampler):
    '''Test-only dataloader.'''
    logger = logging.getLogger('inrlog')
    logger.info(f"==>> dataset_opt['phase']: {dataset_opt['phase']}")

    if opt['dist']:
        print("warning: distributed test is not implemented; test runs on a single GPU.")
        num_workers = dataset_opt['n_workers']
    else:
        num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
    batch_size = dataset_opt['batch_size']

    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, sampler=sampler, drop_last=True,
        pin_memory=True, prefetch_factor=4, persistent_workers=True)


def create_dataset(dataset_opt):
    mode = dataset_opt['mode']
    if mode == 'adobe_ev_from_raw':
        from dataset.adobe_dataset_ev_from_raw import adobeDataset_ev_from_raw as D
    elif mode == 'gopro_ev_from_raw':
        from dataset.gopro_dataset_ev_from_raw import goproDataset_ev_from_raw as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
    dataset = D(dataset_opt)

    logger = logging.getLogger('inrlog')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(
        dataset.__class__.__name__, dataset_opt['mode']))
    return dataset
