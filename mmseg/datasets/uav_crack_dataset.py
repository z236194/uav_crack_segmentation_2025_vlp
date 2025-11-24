from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset


@DATASETS.register_module()
class UAVCrackDataset(BaseSegDataset): #BaseSegDataset
    """UAV-Crack dataset for binary crack segmentation."""

    METAINFO = dict(
        classes=('background', 'crack'),
        palette=[[0, 0, 0], [255, 255, 255]]  # background=black, crack=white
    )

    def __init__(self, **kwargs):
        super().__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs
        )
