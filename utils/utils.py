import importlib
import numpy as np
import cv2
import torch
import torch.distributed as dist


def count_params(model, verbose=False):
    """
    返回params中的参数总数
    params架构：params_groups>model_params_group>params
    """
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params


def check_istarget(name, para_list):
    """ 
    name: full name of source para
    para_list: partial name of target para 
    """
    istarget=False
    for para in para_list:
        if para in name:
            return True
    return istarget


def instantiate_from_config(config):
    """
    config是键值对,target为一个键，其值应该是一个字符串写着 module.cls
    """
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        """这两行代码通常用于开发时动态地加载或更新模块，以便快速地看到代码变化的效果，而无需重启 Python 解释器"""
        module_imp = importlib.import_module(module) # 根据名称导入模块赋值给module_imp
        importlib.reload(module_imp) # 重新在内存中加载
    # 相当于from module import cls
    return getattr(importlib.import_module(module, package=None), cls)


def load_npz_from_dir(data_dir):
    # data (num_data_name, data.shape[0],data.shape[1], ..., data.shape[n])
    # 从 NumPy 文件加载键名为 'arr_0' 的数组, 假如没有键名，这个值为默认值
    data = [np.load(os.path.join(data_dir, data_name))['arr_0'] for data_name in os.listdir(data_dir)]
    # data (num_data_name*data.shape[0], data.shape[1], ..., data.shape[n])
    data = np.concatenate(data, axis=0)
    return data


def load_npz_from_paths(data_paths):
    data = [np.load(data_path)['arr_0'] for data_path in data_paths]
    data = np.concatenate(data, axis=0)
    return data   


def resize_numpy_image(image, max_resolution=512 * 512, resize_short_edge=None):
    h, w = image.shape[:2]
    if resize_short_edge is not None:
        # 使得较短边变成resize_short_edge,宽和高比例如前
        k = resize_short_edge / min(h, w)
    else:
        # 归一化图像使得新的图片的面积为max_resolution
        k = max_resolution / (h * w)
        k = k**0.5
    h = int(np.round(h * k / 64)) * 64
    w = int(np.round(w * k / 64)) * 64
    # 插值使得保有良好的性质
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return image


def setup_dist(args):
    if dist.is_initialized():
        return
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://'
    )
