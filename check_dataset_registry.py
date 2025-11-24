from mmseg.registry import DATASETS
import mmseg.datasets.uav_crack_dataset  # 必须触发 import

print(DATASETS.get('UAVCrackDataset'))
