import math
from inspect import isfunction
import torch
from torch import nn
import torch.distributed as dist


def gather_data(data, return_np=True):
    ''' gather data from multiple processes to one list '''
    '''将分布式进程中的数据都聚集到一个张量中'''
    data_list = [torch.zeros_like(data) for _ in range(dist.get_world_size())] #先创建一个(num_processes,(data))的张量
    dist.all_gather(data_list, data)  # gather not supported with NCCL
    if return_np: # 如果要返回np形式
        data_list = [data.cpu().numpy() for data in data_list]
    return data_list

def autocast(f):
    def do_autocast(*args, **kwargs):
        with torch.cuda.amp.autocast(enabled=True,
                                     dtype=torch.get_autocast_gpu_dtype(),
                                     cache_enabled=torch.is_autocast_cache_enabled()):
            return f(*args, **kwargs)
    return do_autocast


def extract_into_tensor(a, t, x_shape):
    '''a输入张量，t索引张量，x_shape输出张量的形状'''
    # 除了t的最后一维，张量t的前面维度与a的对应前面维度相同
    b, *_ = t.shape
    out = a.gather(-1, t) #在输入张量的最后一个维度按照索引张量t进行提取
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
                                                                    #第一个维度重复次数 #后面维度重复次数都是1，即除了第一个维度后面都不重复
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def exists(val):
    return val is not None

def identity(*args, **kwargs):
    return nn.Identity()

def uniq(arr):
    return{el: True for el in arr}.keys()

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    #                           计算除了第一维度的平均值
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def ismap(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)

def isimage(x):
    if not isinstance(x,torch.Tensor):
        return False
    # (num_batchs,h,w,c)//(num_batchs,c,h,w)
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def shape_to_str(x):
    shape_str = "x".join([str(x) for x in x.shape])
    return shape_str

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

ckpt = torch.utils.checkpoint.checkpoint
def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    '''接受四个参数：func（要评估的函数）、inputs（传递给func的参数序列）、
    params（func依赖但不显式作为参数接受的参数序列）和flag（如果为False，禁用梯度检查点）'''
    if flag:
        return ckpt(func, *inputs)
    else:
        return func(*inputs)

