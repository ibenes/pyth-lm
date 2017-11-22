import torch

class HiddenStateReorganizer():
    def __init__(self, h0_provider):
        self._h0_provider = h0_provider

    def __call__(self, last_h, mask, batch_size):
        if len(mask.size()) == 0:
            return self._h0_provider.init_hidden(batch_size)

        if mask.size(0) > batch_size:
            raise ValueError("Cannot reorganize mask {} to batch size {}".format(mask, batch_size))
        reorg = [h[mask] for h in last_h]

        if mask.size(0) < batch_size:
            additional_h = self._h0_provider.init_hidden(batch_size - mask.size(0))
            reorg = [torch.cat([r_h, a_h]) for r_h, a_h in zip(reorg, additional_h)]

        return tuple(reorg)