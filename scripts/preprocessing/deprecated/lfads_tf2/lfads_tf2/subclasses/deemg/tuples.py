from collections import namedtuple

"""
Classes that organize groups of tensors into namedtuples. 
Names and uses should be self-explanatory.
"""

# modified batch input object for deEMG when applying temporal shift
BatchInput = namedtuple(
    'TshiftBatchInput', [
        'data', # emg data (input to encoders), BxTxDATA_DIM
        'nll_data', # emg data (recon. cost eval), BxTxDATA_DIM
        'sv_mask', # sample validation mask, BxTxDATA_DIM
        'nll_sv_mask', # sample validation mask, BxTxDATA_DIM
        'ext_input', # external inputs, BxTxEXT_INPUT_DIM
    ])

deEMGOutput = namedtuple(
    'deEMGOutput', [
        'rates', # mean estimates, BxTxDATA_DIM
        'alpha', # alpha estimates, BxTxDATA_DIM
        'beta', # beta estimates, BxTxDATA_DIM
        'ic_means', # means for the IC distributions, BxIC_DIM
        'ic_stddevs', # stddev for the IC distributions, BxIC_DIM
        'co_means', # controller output means, BxTxCO_DIM
        'co_stddevs', # controller output stddevs, BxTxCO_DIM
        'factors', # latent factors produced by generator, BxTxFAC_DIM
        'gen_states', # states of the generator RNN, BxTxGEN_DIM
        'gen_init', # initial states of the generator RNN, BxGEN_DIM
        'gen_inputs', # actual inputs to the generator, BxTxCO_DIM
        'con_states', # states of the controller RNN, BxTxCON_DIM
    ])

SamplingOutput = namedtuple(
    'SamplingOutput', [
        'rates', # mean estimates, BxTxDATA_DIM
        'alpha', # alpha estimates, BxTxDATA_DIM
        'beta', # beta estimates, BxTxDATA_DIM
        'factors', # latent factors produced by generator, BxTxFAC_DIM
        'gen_states', # states of the generator RNN, BxTxGEN_DIM
        'gen_inputs', # actual inputs to the generator, BxTxCO_DIM
        'gen_init', # initial states of the generator RNN, BxGEN_DIM
        'ic_post_mean', # means for the IC posterior, BxIC_DIM
        'ic_post_logvar', # log-variance for the IC posterior, BxIC_DIM
        'ic_prior_mean', # means for the IC prior, BxIC_DIM
        'ic_prior_logvar', # log-variance for the IC prior, BxIC_DIM
    ])
