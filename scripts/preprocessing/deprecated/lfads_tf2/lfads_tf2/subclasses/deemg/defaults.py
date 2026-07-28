from lfads_tf2.defaults import get_cfg_defaults as get_base_cfg

def get_cfg_defaults():
    """Get default YACS config node for deEMG"""
    _C = get_base_cfg()
    _C.MODEL.DATA_DIM = _C.MODEL.ENC_INPUT_DIM = 12 # The dimension of the output distribution (i.e. number of neurons).
    _C.MODEL.GEN_DIM = 64 # The hidden dimension of the generator GRUCell.
    _C.MODEL.CON_DIM = 64 # The hidden dimension of the controller GRUCell.
    _C.MODEL.IC_DIM = 30 # The dimension of the initial condition distributions.
    _C.MODEL.CO_DIM = 3 # The dimension of the controller output.
    _C.MODEL.FAC_DIM = 10 # The dimension of the learned factors.
    _C.MODEL.CD_RATE = 0.5 # Rate of samples dropped at the input for CD
    # Add subclass-specific hyperparameters
    _C.TRAIN.DATA.AUGMENT.TEMPORAL_SHIFT = 0 # max number of bins that each channel will be shifted during training (int value)
    _C.TRAIN.DATA.AUGMENT.TEMPORAL_SHIFT_DIST = 'normal' # str specifying type of distribution to randomly sample from for shifts for augmentation
    _C.TRAIN.LR.INIT = 0.002 # the initial learning rate
    return _C.clone()
