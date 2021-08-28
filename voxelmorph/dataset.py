#!/user/bin/env python
# -*- coding:utf-8 -*-

from .generators import *
from torch.utils.data.dataset import Dataset
import torch

def ED_to_Es(ED_names,ES_names, bidir=False, batch_size=1, no_warp=False, **kwargs):
    """
    Generator for scan-to-scan registration.

    Parameters:
        vol_names: List of volume files to load, or list of preloaded volumes.
        bidir: Yield input image as output for bidirectional models. Default is False.
        batch_size: Batch size. Default is 1.
        prob_same: Induced probability that source and target inputs are the same. Default is 0.
        no_warp: Excludes null warp in output list if set to True (for affine training).
            Default if False.
        kwargs: Forwarded to the internal volgen generator.
    """
    zeros = None
    gen_ED = volgen(ED_names, batch_size=batch_size, **kwargs)
    gen_ES = volgen(ES_names, batch_size=batch_size, **kwargs)
    while True:
        scan_ED = next(gen_ED)[0]
        scan_ES = next(gen_ES)[0]

        # some induced chance of making source and target equal
        # if prob_same > 0 and np.random.rand() < prob_same:
        #     if np.random.rand() > 0.5:
        #         scan1 = scan2
        #     else:
        #         scan2 = scan1

        # cache zeros
        if not no_warp and zeros is None:
            shape = scan_ED.shape[1:-1]
            zeros = np.zeros((batch_size, *shape, len(shape)))

        invols = [scan_ED, scan_ES]
        outvols = [scan_ES, scan_ED] if bidir else [scan_ES]
        if not no_warp:
            outvols.append(zeros)

        yield (invols, outvols)

class ED_ES_dataset(Dataset):
    def __init__(self, EDdirs, ESDirs, idxs, bidir =  False):
        super(ED_ES_dataset, self).__init__()
        self.EDdirs = EDdirs
        self.ESdirs = ESDirs
        self.idxs = idxs
        self.bidir = bidir

    def __getitem__(self, item):
        EDdir = self.EDdirs[item]
        ESdir = self.ESdirs[item]
        ED = torch.FloatTensor(np.load(EDdir)['vol']).unsqueeze(dim = 0)
        ES = torch.FloatTensor(np.load(ESdir)['vol']).unsqueeze(dim = 0)
        
        inputs = [ED,ES]
        if self.bidir:
            outputs = [ES,ED]
        else:
            outputs = [ES]
        
        return inputs,outputs


    def __len__(self):
        return len(self.idxs)