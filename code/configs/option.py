import os
import os.path as osp
import logging


def parse(opt, is_train=True):

    # export CUDA_VISIBLE_DEVICES
    gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

    # NOTE: the checkpoint path (path.pretrain_model_G) is set in test.py (see --checkpoint).



    # if opt.get('datasets', None) is not None:
    #     if opt['datasets'].get('test', None) is not None:
    #         if opt['datasets']['test'] .get('is_test', None) ==True :

    #             test_dataset=opt['datasets']['test']['mode']
    #             opt['name'] = f'trained_{model_trained_dataset}_test_{test_dataset}_iter{iter}' #* add


    # print(f"==>> opt['path']['pretrain_model_G']: {opt['path']['pretrain_model_G']}")
    # print(f"==>> opt['name']: {opt['name']}")





    opt['is_train'] = is_train
    if opt['distortion'] == 'sr' or opt['distortion'] == 'isr':
        spatial_upscale = opt.get('spatial_upscale', None)  # 字典如果存在键，返回键值，如果不存在，返回None
    # print(f"==>> spatial_upscale: {spatial_upscale}")

    # datasets
    for phase, dataset in opt['datasets'].items():
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        # if opt['distortion'] == 'sr' or opt['distortion'] == 'isr':
        #     dataset['spatial_upscale'] = spatial_upscale
        is_lmdb = False
        if dataset.get('dataroot_GT', None) is not None:
            dataset['dataroot_GT'] = os.path.expanduser(dataset['dataroot_GT'])
            if dataset['dataroot_GT'].endswith('lmdb'):
                is_lmdb = True
        if dataset.get('dataroot_LQ', None) is not None:
            dataset['dataroot_LQ'] = os.path.expanduser(dataset['dataroot_LQ'])
            if dataset['dataroot_LQ'].endswith('lmdb'):
                is_lmdb = True
        dataset['data_type'] = 'lmdb' if is_lmdb else 'img'
        if dataset['mode'].endswith('mc'):  # for memcached
            dataset['data_type'] = 'mc'
            dataset['mode'] = dataset['mode'].replace('_mc', '')

    # path
    for key, path in opt['path'].items():
        if path and key in opt['path'] and key != 'strict_load':
            opt['path'][key] = osp.expanduser(path)
    opt['path']['root'] = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))
    if is_train:
        # experiments_root = os.path.join(opt['path']['root'], 'experiments', opt['name'])
        experiments_root = os.path.join(opt['path']['result_root'], opt['name'])

        opt['path']['experiments_root'] = experiments_root
        # opt['path']['models'] = os.path.join(experiments_root, 'models')
        # opt['path']['training_state'] = os.path.join(experiments_root, 'training_state')
        opt['path']['log'] = experiments_root
        # opt['path']['val_images'] = os.path.join(experiments_root, 'val_images')

        # change some configs for debug mode
        if 'verbose' in opt['name']:
            opt['train']['val_freq'] = 8
            opt['logger']['print_freq'] = 1
            # opt['logger']['save_checkpoint_freq'] = 8
    else:  # test
        test_dataset = opt['datasets']['test']['name']
        results_root = os.path.join(opt['path']['result_root'], opt['name'], f'test_{test_dataset}')
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root

    # network
    if opt['distortion'] == 'sr' or opt['distortion'] == 'isr':
        # opt['network_G']['spatial_upscale'] = spatial_upscale
        opt['network_G']['spatial_upscale'] = None

    return opt


def dict2str(opt, indent_l=1):
    '''dict to string for logger'''
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg


# convert to NoneDict, which return None for missing key.
class NoneDict(dict):
    def __missing__(self, key):
        return None


def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


def check_resume(opt, resume_iter):
    '''Check resume states and pretrain_model paths'''
    logger = logging.getLogger('inrlog')
    if opt['path']['resume_state']:
        if opt['path'].get('pretrain_model_G', None) is not None or opt['path'].get(
                'pretrain_model_D', None) is not None:
            logger.warning('pretrain_model path will be ignored when resuming training.')

        opt['path']['pretrain_model_G'] = osp.join(opt['path']['models'],
                                                   '{}_G.pth'.format(resume_iter))
        logger.info('Set [pretrain_model_G] to ' + opt['path']['pretrain_model_G'])
        if 'gan' in opt['model']:
            opt['path']['pretrain_model_D'] = osp.join(opt['path']['models'],
                                                       '{}_D.pth'.format(resume_iter))
            logger.info('Set [pretrain_model_D] to ' + opt['path']['pretrain_model_D'])