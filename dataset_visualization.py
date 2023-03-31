import fiftyone as fo
import fiftyone.zoo as foz


dataset = foz.load_zoo_dataset(  # 想要看输入参数的解释的话，选择load_zoo_dataset 按 ctrl+q（pycharm IDE中）
    "coco-2017",  # 指定下载coco-2017类型
    splits=["train", "validation"],  # 指定下载验证集
    label_types=["detections"],  # 指定下载目标检测的类型
    classes=["sports ball", "bird", "traffic light", "kite", "cell phone"],  # 指定下载的类别
    # max_samples=10,  # 指定下载符合条件的最大样本数
    only_matching=True,  # 指定仅下载符合条件的图片，即含有猫的图片
    num_workers=16,  # 指定进程数为1
    dataset_dir="../datasets/tiny-coco",  # 指定下载的数据集保存的路径,尽量不要随意更改，这是保存原始图片的路径，
    dataset_name="tiny-coco",  # 指定新下载的数据集的名称,会检测是否已有,不同的dataset 都是指向了同一个原始图像的路径
)
session = fo.launch_app(dataset)
# session.wait()  #