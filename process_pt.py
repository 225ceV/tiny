import torch
from models.yolo import Model   # 定义的model类 从detectionModel继承, cfg参数为模型类型
from copy import deepcopy
from utils.torch_utils import de_parallel

weights = 'yolov5s.pt'
ckpt = torch.load(weights)
# print(type(ckpt['model']))
yolov5s = Model('models/yolov5s.yaml')
csd = ckpt['model'].float().state_dict()
yolov5s.load_state_dict(csd, strict=False)  # load
print(f'Transferred {len(csd)}/{len(yolov5s.state_dict())} items from {weights}')  # report

# load
weights = 'yolov5s6-5l.pt'
ckpt = torch.load(weights)
yolo5s5lold = Model('models/tiny/yolov5s5hold.yaml', ch=3, nc=6)
csd = ckpt['model'].float().state_dict()
yolo5s5lold.load_state_dict(csd, strict=False)  # load
print(f'Transferred {len(csd)}/{len(yolo5s5lold.state_dict())} items from {weights}')  # report

# process
yolo5s5h = Model('models/tiny/yolov5s5h.yaml')
print('========= before process =========')
# print(yolo5s5lold.model[40])
yolo5s5lold.model[40] = yolo5s5h.model[40]  # random
yolo5s5lold.model[0] = yolov5s.model[0]
yolo5s5lold.model[4] = yolov5s.model[4]
yolo5s5lold.model[10] = yolov5s.model[8]
yolo5s5lold.model[11] = yolov5s.model[9]
# print(yolo5s5lold.model[40])

# save
ckpt['model'] = deepcopy(de_parallel(yolo5s5lold)).half()
torch.save(ckpt, 'yolov5s5h_pretrain.pt')
print('========= after  process =========')

# load test
ckpt = torch.load('yolov5s5h_pretrain.pt')
csd = ckpt['model'].float().state_dict()
yolo5s5h.load_state_dict(csd, strict=False)
print(f'Transferred {len(csd)}/{len(yolo5s5h.state_dict())} items from {weights}')  # report

