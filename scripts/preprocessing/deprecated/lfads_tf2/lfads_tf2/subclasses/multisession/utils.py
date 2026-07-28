from os import path
import h5py
from glob import glob

import numpy as np

from lfads_tf2.tuples import SamplingOutput
from lfads_tf2.utils import load_data

import logging
logger = logging.getLogger(__name__)

def load_posterior_averages(model_dir,
                            merge_tv=False,
                            ps_filename='posterior_samples.h5'):
    """Loads posterior sampling output from a file in the `model_dir`.

    This function is used for loading all of the posterior-sampled rates 
    and other outputs from the HDF5 file created by LFADS posterior 
    sampling.

    Parameters
    ----------
    model_dir : str
        The directory of the model to load from.
    merge_tv : bool, optional
        Whether to merge training and validation data, by default False.
    ps_filename : str, optional
        The name of the posterior sampling file to load from, 
        by default 'posterior_samples.h5'.

    Returns
    -------
    lfads_tf2.tuples.SamplingOutput
        A namedtuple with properties corresponding to LFADS outputs. See 
        fields of the tuple at `lfads_tf2.tuples.SamplingOutput` for more 
        details.

    See Also
    --------
    lfads_tf2.models.LFADS.sample_and_average : Performs posterior sampling.

    """
    output_file = path.join(model_dir, ps_filename)
    prefix = path.splitext(ps_filename)[0]
    if not path.isfile(output_file):
        raise FileNotFoundError("No posterior sampling file found.")
    logger.info(f"Loading posterior samples from {output_file}")

    h5_filenames = sorted(glob(path.join(model_dir, prefix + "*")))

    h5dict = {}
    for h5_filename in h5_filenames:
        with h5py.File(h5_filename, 'r') as h5file:
            # open the h5 file in a dictionary
            h5datasets = h5file.keys()
            datasets = np.unique([x.split('lfads_')[-1] for x in h5datasets])
            for ds in datasets:
                h5dict[ds] = {}
                for k in h5datasets: 
                    if ds in k: 
                        h5dict[ds].update({key: h5file[k][key][()] for key in h5file[k].keys()})

    sampling_output = {}
    for ds in h5dict.keys():
        output = {}
        for field in SamplingOutput._fields:
            if merge_tv: 
                if len(h5dict[ds]['train_' + field].shape) > 1:
                    train_inds = h5dict[ds]['train_inds'].astype('int')
                    valid_inds = h5dict[ds]['valid_inds'].astype('int')
                    num_samples = len(train_inds) + len(valid_inds)
                    data_shape = (num_samples,) + h5dict[ds]['train_' + field].shape[1:]

                    combined_data = np.full(
                        data_shape, 
                        np.nan
                    )
                    
                    combined_data[train_inds, :] = h5dict[ds]['train_' + field]
                    combined_data[valid_inds, :] = h5dict[ds]['valid_' + field]
                else: 
                    # for ic_priors
                    combined_data = h5dict[ds]['train_' + field]

                output[field] = combined_data
            else: 
                output[field] = (h5dict[ds]['train_'+field], h5dict[ds]['valid_'+field])

        # organize the posterior sampling data in namedtuples
        if merge_tv:
            sampling_output[ds] = SamplingOutput(**output)
        else:
            train_output = {field: pair[0] for field, pair in output.items()}
            valid_output = {field: pair[1] for field, pair in output.items()}
            sampling_output[ds] = (SamplingOutput(**train_output),
                                   SamplingOutput(**valid_output))

    return sampling_output
