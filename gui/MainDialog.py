import DialogUI
import random
from PyQt5.QtWidgets import QApplication, QFileDialog, QDialog, QAbstractItemView, QTableWidgetItem
from PyQt5 import QtGui
import cv2
import torch
import numpy as np
import argparse
import torch.backends.cudnn as cudnn
import sys
sys.path.append("..")
from utils.augmentations import letterbox
from utils.torch_utils import select_device
from utils.general import non_max_suppression, scale_coords, check_img_size
from utils.plots import plot_one_box
from models.experimental import attempt_load
import time

class MainDialog(QDialog):
    def __init__(self, parent=None):

        # init ui
        super().__init__(parent)
        self.ui = DialogUI.Ui_MainDialog()
        self.ui.setupUi(self)
        # buttons
        self.ui.OpenButton.clicked.connect(self.open_img)
        self.ui.DetectButton.clicked.connect(self.detect_clk)
        # table
        self.ui.table_statistics.setEditTriggers(QAbstractItemView.NoEditTriggers)    # 不可交互
        self.ui.table_statistics.setSelectionBehavior(QAbstractItemView.SelectRows)    # 单次选中行
        self.ui.table_statistics.horizontalHeader().setVisible(False)                  # 隐藏列标题
        self.ui.table_statistics.verticalHeader().setVisible(False)
        # init options
        self.opt = self._options()
        self.device = select_device(self.opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        cudnn.benchmark = True   # improve performance on a singles GPU

        # init detection
        # Load model
        self.model = attempt_load(
            self.opt.weights, device=self.device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.opt.img_size, s=stride)  # check img_size
        if self.half:
            self.model.half()  # to FP16
        # Get cls names and colors
        self.names = self.model.module.names if hasattr(
            self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255)
                        for _ in range(3)] for _ in self.names]

    def open_img(self):
        folder = QFileDialog()
        folder.setStyleSheet("color: blue; background-color: yellow")
        img_name, _ = folder.getOpenFileName(self, "打开图片", f"f{self.opt.source}", "*.jpg;;*.png;;All Files(*)")   # caption dir filer
        if not img_name:
            return
        img = cv2.imread(img_name)
        print(f'MSG: image {img_name} opened')
        showimg = img
        # with torch.no_grad():
        #     pred, img = self._detect_img(img)
        #     self._plot_img(pred, showimg, img)
        showimg = cv2.cvtColor(showimg, cv2.COLOR_BGR2BGRA)
        showimg = cv2.resize(showimg, (640, int(showimg.shape[0]/showimg.shape[1]*640)), interpolation=cv2.INTER_AREA)
        QtImg = QtGui.QImage(showimg.data, showimg.shape[1], showimg.shape[0], QtGui.QImage.Format_RGB32)
        self.ui.label.setPixmap(QtGui.QPixmap.fromImage(QtImg))
        np.save("tempimg.npy", img)


    def detect_clk(self):
        try:
            img = np.load("tempimg.npy")
        except:
            print('MSG: please open image')
            return
        showimg = img
        pred, img = self._detect_img(img)
        self._plot_img(pred, showimg, img)

    def _detect_img(self, img):
        # padding
        img = letterbox(img, new_shape=self.opt.img_size)[0]
        # Convert
        # BGR to RGB, to 3x416x416
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        tik = time.time()
        pred = self.model(img, augment=self.opt.augment)[0]
        tok = time.time()
        inference_time = tok - tik
        # Apply NMS
        tik = time.time()
        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                   agnostic=self.opt.agnostic_nms)
        tok = time.time()
        NMS_time = tok - tik
        # print(pred)
        # write table
        self.ui.table_statistics.setColumnCount(2)
        self.ui.table_statistics.setRowCount(4)
        new_item = QTableWidgetItem('total obj')
        self.ui.table_statistics.setItem(0, 0, new_item)
        new_item = QTableWidgetItem(f'{pred[0].shape[0]}')
        self.ui.table_statistics.setItem(0, 1, new_item)
        new_item = QTableWidgetItem('total time')
        self.ui.table_statistics.setItem(1, 0, new_item)
        new_item = QTableWidgetItem(f'{inference_time+NMS_time:.2f}sec')
        self.ui.table_statistics.setItem(1, 1, new_item)
        new_item = QTableWidgetItem('inference time')
        self.ui.table_statistics.setItem(2, 0, new_item)
        new_item = QTableWidgetItem(f'{inference_time:.2f}sec')
        self.ui.table_statistics.setItem(2, 1, new_item)
        new_item = QTableWidgetItem('NMS time')
        self.ui.table_statistics.setItem(3, 0, new_item)
        new_item = QTableWidgetItem(f'{NMS_time:.2f}sec')
        self.ui.table_statistics.setItem(3, 1, new_item)
        return pred, img

    def _plot_img(self, pred, showimg, img):
        name_list = []
        # plot
        for i, det in enumerate(pred):
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], showimg.shape).round()   # 参考图片大小缩放坐标

                for *xyxy, conf, cls in reversed(det):          # *xyxy 缺省
                    label = '%s %.2f' % (self.names[int(cls)], conf)
                    name_list.append(self.names[int(cls)])
                    plot_one_box(xyxy, showimg, label=label,
                                 color=self.colors[int(cls)], line_thickness=2)

        cv2.imwrite('prediction.jpg', showimg)   # save results
        # show on ui label
        result = cv2.cvtColor(showimg, cv2.COLOR_BGR2BGRA)
        result = cv2.resize(result, (640, int(result.shape[0]/result.shape[1]*640)), interpolation=cv2.INTER_AREA)
        QtImg = QtGui.QImage(result.data, result.shape[1], result.shape[0], QtGui.QImage.Format_RGB32)
        self.ui.label.setPixmap(QtGui.QPixmap.fromImage(QtImg))

    def _options(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str,
                            default='../weights/wheat-det/yolov5l-p2-cbam.pt', help='model.pt path(s)')
        # file/folder, 0 for webcam
        parser.add_argument('--source', type=str,
                            default='../test_img/', help='source')
        parser.add_argument('--img-size', type=int,
                            default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float,
                            default=0.25, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float,
                            default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='',
                            help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true',
                            help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true',
                            help='save confidences in --save-txt labels')
        parser.add_argument('--nosave', action='store_true',
                            help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int,
                            help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument(
            '--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true',
                            help='augmented inference')
        parser.add_argument('--update', action='store_true',
                            help='update all models')
        parser.add_argument('--project', default='runs/detect',
                            help='save results to project/name')
        parser.add_argument('--name', default='exp',
                            help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true',
                            help='existing project/name ok, do not increment')
        opt = parser.parse_args()
        print(opt)
        return opt


if __name__ == '__main__':
    myapp = QApplication(sys.argv)
    myDlg = MainDialog()
    myDlg.show()
    sys.exit(myapp.exec_())
