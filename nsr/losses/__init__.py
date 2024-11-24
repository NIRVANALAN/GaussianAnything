# 2d reconstruction losses
from .id_loss import IDLoss
# from .lms import HeatmapLoss # for faces
# from .lpips_deprecated.lpips import LPIPS

# manage import
__all__ = [
    # 'LPIPS',
    'IDLoss',
]
