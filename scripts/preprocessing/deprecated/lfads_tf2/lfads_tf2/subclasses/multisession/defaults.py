from lfads_tf2.defaults import get_cfg_defaults as get_base_cfg

def get_cfg_defaults():
    """Get default YACS config node for DimReducedLFADS"""
    _C = get_base_cfg()
    # Add subclass-specific hyperparameters
    _C.MODEL.ENC_INPUT_DIM = 50 # Dimensionality of the spike projection to feed into LFADS for all datasets
    _C.TRAIN.FIX_READIN = True # True if read-in matrices should be frozen and not trained  

    return _C.clone()
