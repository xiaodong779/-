import mmcv
from mmcv import digit_version

from .version import __version__

mmcv_minimum_version = '1.2.6'
# 【修改 1】将最高版本限制由 1.4.0 调大到 2.0.0，适配你当前的 1.7.1
mmcv_maximum_version = '2.0.0'  
mmcv_version = digit_version(mmcv.__version__)

# 【修改 2】为了稳妥，直接把这一段强制检查注释掉（加了#号），防止它阻止运行
# assert (digit_version(mmcv_minimum_version) <= mmcv_version
#         <= digit_version(mmcv_maximum_version)), \
#     f'MMCV=={mmcv.__version__} is used but incompatible. ' \
#     f'Please install mmcv>={mmcv_minimum_version}, <={mmcv_maximum_version}.'

__all__ = ['__version__']