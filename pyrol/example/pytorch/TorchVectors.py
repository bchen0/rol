import collections

from pyrol.pyrol import ROL
from pyrol.getTypeName import *
import torch


class PythonVector(getTypeName('Vector')):

    def __iadd__(self, other):
        self.plus(other)

    # TO-DO: Add other operations.


class TensorVector(PythonVector):

    @torch.no_grad()
    def __init__(self, tensor):
        super().__init__()
        assert isinstance(tensor, torch.Tensor)
        self.torch_object = tensor

    @property
    def tensor(self):
        return self.torch_object

    @torch.no_grad()
    def axpy(self, alpha, other):
        self.tensor.add_(other.tensor, alpha=alpha)

    @torch.no_grad()
    def scale(self, alpha):
        self.tensor.mul_(alpha)

    @torch.no_grad()
    def zero(self):
        self.tensor.zero_()

    @torch.no_grad()
    def dot(self, other):
        ans = torch.sum(torch.mul(self.tensor, other.tensor))
        return ans.item()

    # @torch.no_grad()
    # def clone(self):
    #     tensor = copy.deepcopy(self.tensor.detach())
    #     ans = TensorVector(tensor)
    #     ans.zero()
    #     return ans
    @torch.no_grad()
    def clone(self):
        return TensorVector(torch.zeros_like(self.tensor))

    @torch.no_grad()
    def dimension(self):
        return self.tensor.numel()

    @torch.no_grad()
    def setScalar(self, alpha):
        self.tensor.fill_(alpha)

    @torch.no_grad()
    def __getitem__(self, index):
        flat = self.tensor.view(-1)
        return flat[index].item()

    @torch.no_grad()
    def __setitem__(self, index, value):
        flat = self.tensor.view(-1)
        flat[index] = value

    # Derived methods #########################################################

    @torch.no_grad()
    def reduce(self, op):
        reduction_type = op.reductionType()
        match reduction_type:
            case ROL.Elementwise.REDUCE_MIN:
                ans = self.tensor.min()
                ans = ans.item()
            case ROL.Elementwise.REDUCE_MAX:
                ans = self.tensor.max()
                ans = ans.item()
            case ROL.Elementwise.REDUCE_SUM:
                ans = torch.sum(self.tensor)
                ans = ans.item()
            case ROL.Elementwise.REDUCE_AND:
                ans = self.tensor.all()
                ans = ans.item()
            case ROL.Elementwise.REDUCE_BOR:
                ans = 0
                for i in range(self.dimension()):
                    ans = ans | int(self[i])
            case _:
                raise NotImplementedError(reduction_type)
        return ans

    @torch.no_grad()
    def plus(self, other):
        self.axpy(1, other)

    @torch.no_grad()
    def norm(self):
        return self.dot(self)**0.5

    @torch.no_grad()
    def basis(self, i):
        b = self.clone()
        b.zero()
        b[i] = 1
        return b

    ####

    # @torch.no_grad()
    # def applyUnary(self, op):
    #     for i in range(self.dimension()):
    #         self[i] = op.apply(self[i])
    @torch.no_grad()
    def applyUnary(self, op):
        flat = self.tensor.view(-1)
        for i in range(flat.numel()):
            flat[i] = op.apply(flat[i].item())

    # @torch.no_grad()
    # def applyBinary(self, op, other):
    #     for i in range(self.dimension()):
    #         self[i] = op.apply(self[i], other[i])
    @torch.no_grad()
    def applyBinary(self, op, other):
        flat_self = self.tensor.view(-1)
        flat_other = other.tensor.view(-1)

        for i in range(self.dimension()):
            flat_self[i] = op.apply(flat_self[i].item(), flat_other[i].item())


class TensorDictVector(PythonVector):

    @torch.no_grad()
    def __init__(self, tensor_dict=None, *, flat=None, metadata=None):
        super().__init__()

        if flat is not None:
            assert metadata is not None
            self.flat = flat
            self.metadata = metadata
            self._torch_object = self._make_views(self.flat, self.metadata)
            return

        assert isinstance(tensor_dict, dict)

        metadata = []
        pieces = []
        start = 0

        for k, v in tensor_dict.items():
            assert isinstance(v, torch.Tensor)
            n = v.numel()
            end = start + n
            metadata.append((k, tuple(v.shape), start, end))
            pieces.append(v.reshape(-1))
            start = end

        self.flat = torch.cat(pieces)
        self.metadata = metadata
        self._torch_object = self._make_views(self.flat, self.metadata)

    @property
    def torch_object(self):
        return self._torch_object

    @torch_object.setter
    def torch_object(self, value):
        self.copy_from_tensor_dict(value)

    @property
    def tensor_dict(self):
        return self._torch_object
    
    @torch.no_grad()
    def copy_from_tensor_dict(self, source):
        for k, _, _, _ in self.metadata:
            self._torch_object[k].copy_(source[k])

    @staticmethod
    @torch.no_grad()
    def _make_views(flat, metadata):
        tensor_dict = collections.OrderedDict()
        for k, shape, start, end in metadata:
            tensor_dict[k] = flat[start:end].view(shape)
        return tensor_dict

    @torch.no_grad()
    def clone(self):
        return TensorDictVector(
            flat=torch.zeros_like(self.flat),
            metadata=self.metadata,
        )

    @torch.no_grad()
    def axpy(self, alpha, other):
        self.flat.add_(other.flat, alpha=alpha)

    @torch.no_grad()
    def scale(self, alpha):
        self.flat.mul_(alpha)

    @torch.no_grad()
    def zero(self):
        self.flat.zero_()

    @torch.no_grad()
    def dot(self, other):
        return torch.sum(self.flat * other.flat).item()

    @torch.no_grad()
    def dimension(self):
        return self.flat.numel()

    @torch.no_grad()
    def setScalar(self, alpha):
        self.flat.fill_(alpha)

    @torch.no_grad()
    def __getitem__(self, index):
        return self.flat[index].item()

    @torch.no_grad()
    def __setitem__(self, index, value):
        self.flat[index] = value

    @torch.no_grad()
    def plus(self, other):
        self.axpy(1.0, other)

    @torch.no_grad()
    def norm(self):
        return self.dot(self) ** 0.5

    @torch.no_grad()
    def basis(self, i):
        b = self.clone()
        b[i] = 1.0
        return b

    @torch.no_grad()
    def reduce(self, op):
        reduction_type = op.reductionType()

        match reduction_type:
            case ROL.Elementwise.REDUCE_MIN:
                return self.flat.min().item()

            case ROL.Elementwise.REDUCE_MAX:
                return self.flat.max().item()

            case ROL.Elementwise.REDUCE_SUM:
                return self.flat.sum().item()

            case ROL.Elementwise.REDUCE_AND:
                return self.flat.all().item()

            case ROL.Elementwise.REDUCE_BOR:
                # Fallback. Usually not performance-critical unless called often.
                ans = 0
                flat_cpu = self.flat.detach().cpu().view(-1)
                for i in range(flat_cpu.numel()):
                    ans = ans | int(flat_cpu[i].item())
                return ans

            case _:
                raise NotImplementedError(reduction_type)

    @torch.no_grad()
    def applyUnary(self, op):
        flat = self.flat.view(-1)
        for i in range(flat.numel()):
            flat[i] = op.apply(flat[i].item())

    @torch.no_grad()
    def applyBinary(self, op, other):
        flat_self = self.flat.view(-1)
        flat_other = other.flat.view(-1)

        for i in range(flat_self.numel()):
            flat_self[i] = op.apply(flat_self[i].item(), flat_other[i].item())
