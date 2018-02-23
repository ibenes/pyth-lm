import torch
import IPython

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class InfiniNoneType(metaclass=Singleton):
    def __eq__(self, other):
        return other is None or isinstance(other, InfiniNone)

    def __iter__(self):
        while True:
            yield InfiniNoneType()

InfiniNone = InfiniNoneType()

def reorg_single(last_h, mask, additional_h=None):
    reorg = last_h[:, mask]
    if additional_h is not InfiniNone:
        reorg = torch.cat([reorg, additional_h], dim=1)

    return reorg


class TensorReorganizer():
    def __init__(self, h0_provider):
        self._h0_provider = h0_provider

    def __call__(self, last_h, mask, batch_size):
        if len(mask.size()) == 0:
            return self._h0_provider(batch_size)

        if mask.size(0) > batch_size:
            raise ValueError("Cannot reorganize mask {} to batch size {}".format(mask, batch_size))

        if isinstance(last_h, tuple):
            single_var = False
        elif isinstance(last_h, torch._TensorBase) or (last_h, torch.autograd.Variable):
            single_var = True
        else:
            raise TypeError(
                "last_h has unsupported type {}, "
                "only tuples, Tensors, and Variables are accepted".format(
                        last_h.__class__)
            )

        adding = mask.size(0) < batch_size
        if adding:
            nb_needed_h0 = batch_size - mask.size(0)
            additional_h = self._h0_provider(nb_needed_h0)
        else:
            additional_h = InfiniNone

        if single_var:
            reorg = reorg_single(last_h, mask, additional_h)
        else:
            reorg = tuple(reorg_single(h, mask, a_h) for h, a_h in zip(last_h, additional_h))

        return reorg
