import collections
import math

from pyrol.pyrol import ROL
from pyrol.getTypeName import *
import torch


def _op_name(op):
    return type(op).__name__


def _unary_parameter(op, probe):
    return float(op.apply(float(probe)))


def _fill_unary_(tensor, op):
    tensor.fill_(_unary_parameter(op, 0.0))


def _shift_unary_(tensor, op):
    tensor.add_(_unary_parameter(op, 0.0))


def _scale_unary_(tensor, op):
    tensor.mul_(_unary_parameter(op, 1.0))


def _power_unary_(tensor, op):
    value = _unary_parameter(op, 2.0)
    if value <= 0.0:
        return False
    tensor.pow_(math.log(value, 2.0))
    return True


def _round_unary_(tensor, op):
    ceil = torch.ceil(tensor)
    floor = torch.floor(tensor)
    tensor.copy_(torch.where(ceil - tensor <= 0.5, ceil, floor))


def _log_unary_(tensor, op):
    ninf = -0.1 * torch.finfo(tensor.dtype).max
    tensor.copy_(torch.where(tensor > 0, torch.log(tensor), torch.full_like(tensor, ninf)))


def _heaviside_unary_(tensor, op):
    tensor.copy_(torch.where(tensor > 0,
                             torch.ones_like(tensor),
                             torch.where(tensor == 0,
                                         torch.full_like(tensor, 0.5),
                                         torch.zeros_like(tensor))))


def _threshold_upper_unary_(tensor, op):
    threshold = _unary_parameter(op, -1.0e100)
    tensor.clamp_min_(threshold)


def _threshold_lower_unary_(tensor, op):
    threshold = _unary_parameter(op, 1.0e100)
    tensor.clamp_max_(threshold)


def _build_c_unary_(tensor, op):
    tensor.mul_(0.5)
    tensor.clamp_max_(1.0)


_UNARY_HANDLERS = {
    "AbsoluteValue_double_t": lambda tensor, op: tensor.abs_(),
    "BuildC": _build_c_unary_,
    "Fill_double_t": _fill_unary_,
    "Heaviside_double_t": _heaviside_unary_,
    "Logarithm_double_t": _log_unary_,
    "Power_double_t": _power_unary_,
    "Reciprocal_double_t": lambda tensor, op: tensor.reciprocal_(),
    "Round_double_t": _round_unary_,
    "Scale_double_t": _scale_unary_,
    "Shift_double_t": _shift_unary_,
    "Sign_double_t": lambda tensor, op: tensor.copy_(torch.sign(tensor)),
    "SquareRoot_double_t": lambda tensor, op: tensor.sqrt_(),
    "ThresholdLower_double_t": _threshold_lower_unary_,
    "ThresholdUpper_double_t": _threshold_upper_unary_,
}


def _axpy_binary_(tensor, other, op):
    tensor.add_(other, alpha=float(op.apply(0.0, 1.0)))


def _aypx_binary_(tensor, other, op):
    tensor.mul_(float(op.apply(1.0, 0.0)))
    tensor.add_(other)


def _active_binary_(tensor, other, op):
    # This is exact for the offsets used by ROL_Bounds' scaling code. Other
    # offsets fall back to the scalar path because the threshold is private.
    eps = 1.0e-12
    candidates = (-2.0, 0.0)
    for offset in candidates:
        if (op.apply(1.0, offset) == 0.0
                and op.apply(1.0, offset + eps) == 1.0
                and op.apply(1.0, offset - eps) == 0.0):
            tensor.masked_fill_(other <= offset, 0.0)
            return True
    return False


def _prune_binding_binary_(tensor, other, op):
    tensor.mul_((other == 1).to(dtype=tensor.dtype))


def _set_zero_entry_binary_(tensor, other, op):
    tensor.copy_(torch.where(tensor == 0, other, tensor))


_BINARY_HANDLERS = {
    "Active": _active_binary_,
    "Aypx_double_t": _aypx_binary_,
    "Axpy_double_t": _axpy_binary_,
    "Divide_double_t": lambda tensor, other, op: tensor.div_(other),
    "DivideAndInvert_double_t": lambda tensor, other, op: tensor.copy_(other / tensor),
    "Greater": lambda tensor, other, op: tensor.copy_(torch.maximum(tensor, other)),
    "Greater_double_t": lambda tensor, other, op: tensor.copy_(torch.maximum(tensor, other)),
    "Lesser": lambda tensor, other, op: tensor.copy_(torch.minimum(tensor, other)),
    "Lesser_double_t": lambda tensor, other, op: tensor.copy_(torch.minimum(tensor, other)),
    "LowerBinding": None,
    "Max_double_t": lambda tensor, other, op: tensor.copy_(torch.maximum(tensor, other)),
    "Min_double_t": lambda tensor, other, op: tensor.copy_(torch.minimum(tensor, other)),
    "Multiply_double_t": lambda tensor, other, op: tensor.mul_(other),
    "Plus_double_t": lambda tensor, other, op: tensor.add_(other),
    "PruneBinding": _prune_binding_binary_,
    "Set_double_t": lambda tensor, other, op: tensor.copy_(other),
    "SetZeroEntry": _set_zero_entry_binary_,
    "UpperBinding": None,
    "isGreater": lambda tensor, other, op: tensor.copy_((tensor > other).to(dtype=tensor.dtype)),
}


def _apply_unary_fast_(tensor, op):
    handler = _UNARY_HANDLERS.get(_op_name(op))
    if handler is None:
        return False
    result = handler(tensor, op)
    return result is not False


def _apply_binary_fast_(tensor, other, op):
    handler = _BINARY_HANDLERS.get(_op_name(op))
    if handler is None:
        return False
    result = handler(tensor, other, op)
    return result is not False


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
        if _apply_unary_fast_(self.tensor, op):
            return
        flat = self.tensor.view(-1)
        for i in range(flat.numel()):
            flat[i] = op.apply(flat[i].item())

    # @torch.no_grad()
    # def applyBinary(self, op, other):
    #     for i in range(self.dimension()):
    #         self[i] = op.apply(self[i], other[i])
    @torch.no_grad()
    def applyBinary(self, op, other):
        if _apply_binary_fast_(self.tensor, other.tensor, op):
            return
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
        if _apply_unary_fast_(self.flat, op):
            return
        flat = self.flat.view(-1)
        for i in range(flat.numel()):
            flat[i] = op.apply(flat[i].item())

    @torch.no_grad()
    def applyBinary(self, op, other):
        if _apply_binary_fast_(self.flat, other.flat, op):
            return
        flat_self = self.flat.view(-1)
        flat_other = other.flat.view(-1)

        for i in range(flat_self.numel()):
            flat_self[i] = op.apply(flat_self[i].item(), flat_other[i].item())
