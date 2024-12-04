import logging
from itertools import chain
import pdb
import os
import shutil

import torch
import numpy as np

from torch_geometric.data import Batch
from utils import add_errors_as_bfactors, atom37_to_pdb
from refine import refine


class DataParallel(torch.nn.DataParallel):
    r"""Implements data parallelism at the module level.

    This container parallelizes the application of the given :attr:`module` by
    splitting a list of :class:`torch_geometric.data.Data` objects and copying
    them as :class:`torch_geometric.data.Batch` objects to each device.
    In the forward pass, the module is replicated on each device, and each
    replica handles a portion of the input.
    During the backwards pass, gradients from each replica are summed into the
    original module.

    The batch size should be larger than the number of GPUs used.

    The parallelized :attr:`module` must have its parameters and buffers on
    :obj:`device_ids[0]`.

    .. note::

        You need to use the :class:`torch_geometric.loader.DataListLoader` for
        this module.

    Args:
        module (Module): Module to be parallelized.
        device_ids (list of int or torch.device): CUDA devices.
            (default: all devices)
        output_device (int or torch.device): Device location of output.
            (default: :obj:`device_ids[0]`)
        follow_batch (list or tuple, optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (list or tuple, optional): Will exclude each key in the
            list. (default: :obj:`None`)
    """
    def __init__(self, module, device_ids=None, output_device=None,
                 follow_batch=None, exclude_keys=None):
        super().__init__(module, device_ids, output_device)
        self.src_device = torch.device(f'cuda:{self.device_ids[0]}')
        self.follow_batch = follow_batch or []
        self.exclude_keys = exclude_keys or []
        self.replicates = self.replicate_model() # This ensures that model is only copied to the devices once

    def replicate_model(self):
        
        return self.replicate(self.module, self.device_ids)

    def forward(self, data_list):
        """"""
        if len(data_list) == 0:
            logging.warning('DataParallel received an empty data list, which '
                            'may result in unexpected behavior.')
            return None

        if not self.device_ids or len(self.device_ids) == 1:  # Fallback
            data = Batch.from_data_list(
                data_list, follow_batch=self.follow_batch,
                exclude_keys=self.exclude_keys).to(self.src_device)
            return self.module(data)

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device:
                raise RuntimeError(
                    f"Module must have its parameters and buffers on device "
                    f"'{self.src_device}' but found one of them on device "
                    f"'{t.device}'")

        inputs = self.scatter(data_list, self.device_ids)
        replicas = self.replicates[:len(inputs)]
        outputs = self.parallel_apply(replicas, inputs, None)
        
        return outputs

    def scatter(self, data_list, device_ids):
        num_devices = min(len(device_ids), len(data_list))

        count = torch.tensor([data.num_nodes for data in data_list])
        cumsum = count.cumsum(0)
        cumsum = torch.cat([cumsum.new_zeros(1), cumsum], dim=0)
        device_id = num_devices * cumsum.to(torch.float) / cumsum[-1].item()
        device_id = (device_id[:-1] + device_id[1:]) / 2.0
        device_id = device_id.to(torch.long)  # round.
        split = device_id.bincount().cumsum(0)
        split = torch.cat([split.new_zeros(1), split], dim=0)
        split = torch.unique(split, sorted=True)
        split = split.tolist()

        return [
            Batch.from_data_list(data_list[split[i]:split[i + 1]],
                                 follow_batch=self.follow_batch,
                                 exclude_keys=self.exclude_keys).to(
                                     torch.device(f'cuda:{device_ids[i]}'))
            for i in range(len(split) - 1)
        ]


def find_alignment_transform(traces):
    centers = np.mean(traces, axis=-2, keepdims=True)
    traces = traces - centers

    p1, p2 = traces[0], traces[1:]
    C = np.einsum("i j k, j l -> i k l", p2, p1)
    V, _, W = np.linalg.svd(C)
    U = V @ W 
    U = np.matmul(
        np.stack(
            [
                np.ones(len(p2)),
                np.ones(len(p2)),
                np.linalg.det(U),
            ],
            axis=1,
        )[:, :, None]
        * V,
        W,
    )

    return np.concatenate([np.eye(3)[None], U]), centers

class compute_prediction_error:
    def __init__(self, numbered_sequences, predictions, refine=True):
        
        self.numbered_sequences = numbered_sequences
        self.atoms = [x.cpu().numpy() for x in predictions]
        self.refine = refine
     
        traces = np.stack([x[:, 0] for x in self.atoms])
        self.R, self.t = find_alignment_transform(traces)
        self.aligned_traces = (traces - self.t) @ self.R
        self.error_estimates = (
            np.sum(np.square(self.aligned_traces - np.mean(self.aligned_traces, axis=0)), axis=-1)
        )
        self.ranking = [x.item() for x in np.argsort(np.mean(self.error_estimates, axis=-1))]

    def save_single_unrefined(self, filename, index=0):
        atoms = (self.atoms[index] - self.t[index]) @ self.R[index]
        #         atoms = self.atoms[index]
        unrefined = atom37_to_pdb(self.numbered_sequences, atoms)

        with open(filename, "w+") as file:
            file.write(unrefined)
    
    def save_all(
        self,
        uid,
        dirname=None,
        header=""
    ):
        if dirname is None:
            dirname = "NanoFold_output"
        os.makedirs(dirname, exist_ok=True)

        for i in range(len(self.atoms)):
            unrefined_filename = os.path.join(
                dirname, f"{uid}_rank{self.ranking.index(i)}_unrefined.pdb"
            )
            self.save_single_unrefined(unrefined_filename, index=i)

        np.save(
            os.path.join(dirname, f"{uid}_error_estimates"),
            np.mean(self.error_estimates, axis=0),
        )
        add_errors_as_bfactors(
            os.path.join(dirname, f"{uid}_rank0_unrefined.pdb"), 
            np.sqrt(self.error_estimates.mean(0)), header=[header]
        )