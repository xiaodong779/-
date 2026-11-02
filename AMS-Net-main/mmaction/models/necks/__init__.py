from .tpn import TPN
from .cstp import CSTP
# ✅ 1. 导入你新建的竞争融合颈部类
from .simple_competitive_neck import SimpleCompetitiveNeck

# ✅ 2. 在 __all__ 中登记，使框架能通过配置文件找到它
__all__ = ['TPN', 'CSTP', 'SimpleCompetitiveNeck']