import logging
logger = logging.getLogger('inrlog')


def create_model(opt):
    model = opt['model']
    if model == 'VideoSR_base':
        from .VideoSR_base_model import VideoSRBaseModel as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m

# import models.modules.Sakuya_arch as Sakuya_arch
# model = Sakuya_arch.LunaTokis(64, 6, 8, 5, 40)
