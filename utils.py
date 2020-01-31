import tensorflow as tf
import torch
import numpy as np

from typing import Tuple


def compare_tensors(torch_tensor: torch.Tensor, tf_tensor: tf.Tensor):
    """ Compares a torch and tensorflow tensor and prints the result """
    shape_to = tuple(torch_tensor.shape)
    shape_tf = tf_tensor.shape

    all_close = False
    if shape_to == shape_tf:
        all_close = np.allclose(torch_tensor.cpu().numpy(), tf_tensor.numpy())

    verdict = 'Tensors are identical' if all_close else 'Tensors are NOT identical'

    print("\n{}\nTorch: {}\nTf:    {}".format(verdict, tuple(torch_tensor.shape), tf_tensor.shape))


def get_torch_and_tf_tensor(*dims) -> Tuple[torch.Tensor, tf.Tensor]:
    """ Creates a torch and tensorflow with the given dimension and the same contents """
    t = np.random.rand(*dims)
    return torch.from_numpy(t), tf.convert_to_tensor(t)
