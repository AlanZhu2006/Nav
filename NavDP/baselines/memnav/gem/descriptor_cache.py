"""Incremental device copy of GEM's immutable, append-only CPU descriptors.

The reader still receives the original contiguous [1, frames, width] tensor.
Keeping that shape also preserves the existing cosine reduction: evaluating
only its current row can select a different floating-point reduction kernel.
This cache changes data movement, not retrieval, history eligibility or scores.
"""
import torch


class DescriptorReadCache:
    """One lazily allocated buffer, shared across goal sessions in one episode.

    The owner must reset this cache when replacing its causal history. Already
    published descriptors are immutable. The buffer grows geometrically and
    uploads each descriptor once; it stores no geometry or matching features.
    Calls use the same serialized execution contract as the GEM writer.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._source = None
        self._storage = None
        self._device = None
        self.count = 0
        self.uploaded_bytes = 0
        self.growth_copy_bytes = 0
        self.failure = None

    @torch.no_grad()
    def read(self, descriptors, device):
        if self.failure is not None:
            raise RuntimeError('Descriptor read cache failed; reset is required: ' + self.failure)
        count = len(descriptors)
        device = torch.device(device)
        if count < 1:
            raise ValueError('Descriptor history must be nonempty')
        if self._source is not None and (
                self._source is not descriptors or count < self.count):
            raise ValueError('Descriptor history changed; reset the read cache first')
        if self._device is not None and self._device != device:
            raise ValueError('Descriptor device changed; reset the read cache first')
        if count == self.count:
            return self._storage[:count][None]

        # No access to previously uploaded rows, including after goal switches.
        pending = torch.stack(descriptors[self.count:], 0)
        if (pending.ndim != 2 or pending.shape[1] < 1
                or pending.device.type != 'cpu' or pending.dtype != torch.float32):
            raise ValueError('GEM descriptors must be one-dimensional CPU FP32 rows')
        if self._storage is not None and pending.shape[1] != self._storage.shape[1]:
            raise ValueError('Descriptor width changed within the episode')
        try:
            capacity = 0 if self._storage is None else self._storage.shape[0]
            if count > capacity:
                storage = torch.empty((max(count, 2 * capacity), pending.shape[1]),
                    dtype=pending.dtype, device=device)
                if self.count:
                    storage[:self.count].copy_(self._storage[:self.count])
                storage[self.count:count].copy_(pending)
                self.growth_copy_bytes += self.count * pending.shape[1] * pending.element_size()
                self._storage = storage
            else:
                self._storage[self.count:count].copy_(pending)
            self.uploaded_bytes += pending.numel() * pending.element_size()
            self.count = count
            self._source = descriptors
            self._device = device
        except BaseException as error:
            self.failure = f'{type(error).__name__}: {error}'
            raise
        return self._storage[:count][None]

    def statistics(self):
        width = 0 if self._storage is None else self._storage.shape[1]
        return dict(frames=self.count,
            device=None if self._storage is None else str(self._storage.device),
            capacity_frames=0 if self._storage is None else self._storage.shape[0],
            effective_bytes=self.count * width * 4,
            allocated_bytes=0 if self._storage is None else self._storage.numel() * 4,
            uploaded_bytes=self.uploaded_bytes, growth_copy_bytes=self.growth_copy_bytes,
            failure=self.failure)
