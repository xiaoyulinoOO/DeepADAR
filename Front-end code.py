import sys
import os
import numpy as np
import spectral.io.envi as envi
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, QVBoxLayout, QLabel, QComboBox, QHBoxLayout, QSizePolicy, QSpacerItem, QDialog, QListWidget,QSplitter,QTableWidget,QTableWidgetItem,QMessageBox
from PyQt5.QtGui import QPixmap, QImage, QPen, QPainter
from PyQt5.QtCore import Qt, QPointF
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import cv2
from matplotlib.figure import Figure



class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention_weights = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_output):
        # lstm_output: [batch_size, seq_len, hidden_size]
        scores = self.attention_weights(lstm_output)  # [batch_size, seq_len, 1]
        scores = scores.squeeze(-1)  # [batch_size, seq_len]
        attention_weights = F.softmax(scores, dim=1)  # [batch_size, seq_len]

        # context vector as weighted sum of lstm_output
        context_vector = torch.bmm(attention_weights.unsqueeze(1), lstm_output)
        context_vector = context_vector.squeeze(1)  # [batch_size, hidden_size]
        return context_vector, attention_weights



class ResNet_CNN_LSTM(nn.Module):
    def __init__(self, input_size=256, hidden_size=128, num_layers=1, dropout=0.5):
        super(ResNet_CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=11, padding=5)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.lstm = nn.LSTM(
            input_size=64,  # 调整后的输入通道
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )

        self.attention = Attention(hidden_size)

        # 新增的全连接层
        self.fc_extra = nn.Linear(hidden_size, 128)  # 额外的全连接层

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128, 64)  # 调整后的全连接层
        self.fc2 = nn.Linear(64, 7)  # 输出层

    def forward(self, x):
        # CNN部分
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.permute(0, 2, 1)  # 调整维度以适应LSTM输入

        # LSTM部分
        lstm_out, _ = self.lstm(x)

        # Attention机制
        context_vector, attention_weights = self.attention(lstm_out)

        # 通过新增的全连接层
        x = self.fc_extra(context_vector)
        x = F.relu(x)
        x = self.dropout(x)  # 在此添加Dropout层

        # 通过原来的全连接层
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)  # 再次添加Dropout层
        x = self.fc2(x)

        return x


class SpectrumDialog(QDialog):
    def __init__(self, coordinates, spectra, wavelengths, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Spectrum Viewer')
        self.setGeometry(400, 300, 800, 600)

        # 创建QSplitter来分割坐标和光谱图
        splitter = QSplitter(Qt.Horizontal)

        # 创建坐标显示区域
        self.coord_widget = QWidget()
        coord_layout = QVBoxLayout()
        self.coord_table = QTableWidget()
        self.coord_table.setColumnCount(1)
        self.coord_table.setHorizontalHeaderLabels(['Coordinates'])
        self.coord_table.verticalHeader().hide()

        # 填充坐标表格
        for i, coord in enumerate(coordinates):
            self.coord_table.insertRow(i)
            self.coord_table.setItem(i, 0, QTableWidgetItem(coord))

        coord_layout.addWidget(self.coord_table)
        self.coord_widget.setLayout(coord_layout)

        # 创建光谱图显示区域
        self.spectrum_widget = QWidget()
        spectrum_layout = QVBoxLayout()

        # 使用 Matplotlib 绘制光谱图
        self.figure = Figure(figsize=(8, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.subplots()
        self.axes.set_title('Spectra')
        self.axes.set_xlabel('Wavelength (nm)')
        self.axes.set_ylabel('Intensity')

        # 初始化光谱图
        self.plot_all_spectra(spectra, wavelengths)

        spectrum_layout.addWidget(self.canvas)
        self.spectrum_widget.setLayout(spectrum_layout)

        # 将坐标和光谱图部件添加到splitter中
        splitter.addWidget(self.coord_widget)
        splitter.addWidget(self.spectrum_widget)

        # 设置splitter的初始尺寸比例
        splitter.setSizes([splitter.width() // 3, splitter.width() * 2 // 3])

        # 设置弹窗的布局
        layout = QVBoxLayout()
        layout.addWidget(splitter)
        self.setLayout(layout)

    def plot_all_spectra(self, spectra, wavelengths):
        self.axes.clear()
        self.axes.set_title('Spectra')
        self.axes.set_xlabel('Wavelength (nm)')
        self.axes.set_ylabel('Intensity')

        for i, (spectrum, wavelength) in enumerate(zip(spectra, wavelengths)):
            self.axes.plot(wavelength, spectrum, label=f'Point {i + 1}')

        self.axes.legend()
        self.canvas.draw()


class SaveDialog(QDialog):
    def __init__(self, original_image, predicted_image, parent=None):
        super().__init__(parent)
        self.setWindowTitle('图片保存')

        # 获取主窗口的位置和大小
        if parent:
            parent_rect = parent.geometry()
            x = parent_rect.x() + parent_rect.width() + 10 # 放置在主窗口右侧
            y = parent_rect.y() + (parent_rect.height() - 150) // 2  # 垂直居中
        else:
            x = 300  # 默认位置
            y = 200

        self.setGeometry(x, y, 400, 300)  # 设置对话框的位置和大小

        layout = QVBoxLayout()

        self.save_original_button = QPushButton('保存原始图像')
        self.save_predicted_button = QPushButton('保存预测图像')
        self.save_both_button = QPushButton('两张图片都进行保存')
        self.cancel_button = QPushButton('不进行保存')

        layout.addWidget(self.save_original_button)
        layout.addWidget(self.save_predicted_button)
        layout.addWidget(self.save_both_button)
        layout.addWidget(self.cancel_button)

        self.setLayout(layout)

        self.original_image = original_image
        self.predicted_image = predicted_image

        # 连接信号和槽
        self.save_original_button.clicked.connect(self.save_original)
        self.save_predicted_button.clicked.connect(self.save_predicted)
        self.save_both_button.clicked.connect(self.save_both)
        self.cancel_button.clicked.connect(self.reject)

    def save_original(self):
        # 保存原图
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Original Image", "", "PNG Files (*.png);;All Files (*)",
                                                   options=options)
        if file_path:
            self.original_image.save(file_path, format='PNG')
        self.accept()

    def save_predicted(self):
        # 保存预测图
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Predicted Image", "", "PNG Files (*.png);;All Files (*)",
                                                   options=options)
        if file_path:
            self.predicted_image.save(file_path, format='PNG')
        self.accept()

    def save_both(self):
        # 保存原图和预测图
        options = QFileDialog.Options()
        original_path, _ = QFileDialog.getSaveFileName(self, "Save Original Image", "",
                                                       "PNG Files (*.png);;All Files (*)",
                                                       options=options)
        if original_path:
            self.original_image.save(original_path, format='PNG')

        predicted_path, _ = QFileDialog.getSaveFileName(self, "Save Predicted Image", "",
                                                        "PNG Files (*.png);;All Files (*)",
                                                        options=options)
        if predicted_path:
            self.predicted_image.save(predicted_path, format='PNG')
        self.accept()
# PyQt5 界面
class AppDemo(QWidget):
    def __init__(self):
        super().__init__()
        # 调整窗口大小以适应扩大后的图像
        self.resize(700, 700)
        self.setWindowTitle('Hyperspectral Image Predictor')
        self.image_label = QLabel(self)


        # 主布局 - 水平布局
        main_layout = QHBoxLayout(self)

        # 左侧布局 - 垂直布局，放置按钮和波长选择
        left_layout = QVBoxLayout()

        # 按钮布局 - 垂直布局
        button_layout = QVBoxLayout()
        self.choose_button = QPushButton('Choose HDR')
        self.generate_button = QPushButton('Generate Image')
        self.predict_button = QPushButton('Predict Image')
        for button in [self.choose_button, self.generate_button, self.predict_button]:
            button_layout.addWidget(button)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # 添加按钮布局到左侧布局
        left_layout.addLayout(button_layout)

        # 添加间隔
        left_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # 波长选择下拉列表
        self.wavelength_combo_r = QComboBox()
        self.wavelength_combo_g = QComboBox()
        self.wavelength_combo_b = QComboBox()
        for combo in [self.wavelength_combo_r, self.wavelength_combo_g, self.wavelength_combo_b]:
            combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # 波长选择布局 - 垂直布局
        wavelength_layout = QVBoxLayout()
        for label, combo in zip(['Red_wavelength:', 'Green_wavelength:', 'Blue_wavelength:'],
                                [self.wavelength_combo_r, self.wavelength_combo_g, self.wavelength_combo_b]):
            wavelength_layout.addWidget(QLabel(label))
            wavelength_layout.addWidget(combo)
            wavelength_layout.addSpacerItem(QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Fixed))

        # 将波长选择布局添加到左侧布局中
        left_layout.addLayout(wavelength_layout)
        left_layout.addStretch(1)

        # 图像布局 - 垂直布局，用于放置原始图像和预测图像
        image_layout = QVBoxLayout()

        # 原始图像标签
        self.original_label = QLabel('原始图像')
        self.original_label.setAlignment(Qt.AlignCenter)
        # 设置固定大小
        self.original_label.setFixedSize(300, 320)
        self.original_label.setScaledContents(True)
        image_layout.addWidget(self.original_label)

        # 预测图像标签
        self.predicted_label = QLabel('预测图像')
        self.predicted_label.setAlignment(Qt.AlignCenter)
        # 设置固定大小
        self.predicted_label.setFixedSize(300, 320)
        self.predicted_label.setScaledContents(True)
        image_layout.addWidget(self.predicted_label)

        # 将左侧布局和图像布局添加到主布局中
        main_layout.addLayout(left_layout)
        main_layout.addLayout(image_layout)

        # 调整伸缩因子
        main_layout.setStretchFactor(left_layout, 0)
        main_layout.setStretchFactor(image_layout, 1)

        self.setLayout(main_layout)

        # 初始化点击点列表
        self.clicked_points = []

        # 连接信号和槽
        self.choose_button.clicked.connect(self.choose_hdr_file)
        self.generate_button.clicked.connect(self.generate_and_show_image)
        self.predict_button.clicked.connect(self.predict_and_show)



        # 为 original_label 添加鼠标点击事件监听
        self.original_label.mousePressEvent = self.on_original_label_click


        # 初始化其他变量
        self.data = None
        self.data_tensor = None
        self.image_size = None
        self.clicked_points = []

        self.wavelengths = None
        self.net = ResNet_CNN_LSTM(input_size=256, hidden_size=128, num_layers=1, dropout=0.5)
        self.load_model()

    def on_original_label_click(self, event):
        if self.data is None:
            return

        # 获取点击位置在标签中的坐标
        pos = event.pos()
        label_width = self.original_label.width()  # 显示宽度 300
        label_height = self.original_label.height()  # 显示高度 320
        image_width = self.image_size[0]  # 原始宽度
        image_height = self.image_size[1]  # 原始高度


        x_ratio = image_width / label_width
        y_ratio = image_height / label_height
        x = int(pos.x() * x_ratio)
        y = int(pos.y() * y_ratio)

        # 检查点击位置是否在图像范围内
        if not (0 <= x < image_width) or not (0 <= y < image_height):
            return

        # 获取点击位置的光谱数据
        spectrum = self.data[y, x, :]
        wavelength = self.wavelengths

        # 检查该点是否已被点击过
        if (x, y) in self.clicked_points:
            self.clicked_points.remove((x, y))
            QMessageBox.information(self, "Information", "Point already clicked. Removing from list.")
        else:
            self.clicked_points.append((x, y))

            # 限制点击点的数量
            if len(self.clicked_points) > 10:
                QMessageBox.information(self, "Information", "Maximum 10 points allowed.")
                return

            # 显示光谱弹窗
            coordinates = [f"({x_}, {y_})" for x_, y_ in self.clicked_points]
            spectra = [self.data[y_, x_, :] for x_, y_ in self.clicked_points]
            dialog = SpectrumDialog(coordinates, spectra, [wavelength] * len(self.clicked_points))
            dialog.exec_()

    def load_model(self):
        # 加载训练好的模型
        model_save_path = 'resnet_lstm_cnn_turning-final_model.pth'
        if os.path.exists(model_save_path):
            self.net.load_state_dict(torch.load(model_save_path))
            self.net.eval()
        else:
            print("Model file not found!")

    def choose_hdr_file(self):
        # 选择 HDR 文件
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Choose HDR File", "", "HDR Files (*.hdr);;All Files (*)",
                                                   options=options)
        if file_path:
            self.load_hdr_file(file_path)

    def load_hdr_file(self, file_path):
        # 加载 HDR 文件
        image = envi.open(file_path)
        self.image_size = [int(image.metadata['samples']), int(image.metadata['lines'])]
        bandnumber = int(image.metadata['bands'])
        self.data = image.read_bands(range(0, bandnumber))

        # 获取波长列表并填充到下拉列表中
        self.wavelengths = np.array(image.metadata['wavelength'], dtype='float')
        self.wavelength_combo_r.clear()
        self.wavelength_combo_g.clear()
        self.wavelength_combo_b.clear()
        for wave in self.wavelengths:
            self.wavelength_combo_r.addItem(str(wave))
            self.wavelength_combo_g.addItem(str(wave))
            self.wavelength_combo_b.addItem(str(wave))

    def generate_and_show_image(self):
        if self.data is not None and self.wavelengths is not None:
            #获取用户选择的波长
            red_wave = float(self.wavelength_combo_r.currentText())
            green_wave = float(self.wavelength_combo_g.currentText())
            blue_wave = float(self.wavelength_combo_b.currentText())

            # 找到最接近的波长索引
            red_band_index = (np.abs(self.wavelengths - red_wave)).argmin()
            green_band_index = (np.abs(self.wavelengths - green_wave)).argmin()
            blue_band_index = (np.abs(self.wavelengths - blue_wave)).argmin()

            # 生成RGB图像
            red_band = self.data[:, :, red_band_index]
            green_band = self.data[:, :, green_band_index]
            blue_band = self.data[:, :, blue_band_index]

            # 保存生成的RGB图像为类的属性，以便在predict_and_show中使用

            rgb_image = np.stack((red_band, green_band, blue_band), axis=-1)
            rgb_image = (rgb_image / rgb_image.max() * 255).astype(np.uint8)

            # 保存生成的RGB图像为类的属性，以便在predict_and_show中使用

            self.rgb_image = Image.fromarray(rgb_image)

            # 对图像进行缩放
            scaled_rgb_image = cv2.resize(rgb_image, (300, 320), interpolation=cv2.INTER_LINEAR)
            qimg = QImage(scaled_rgb_image, scaled_rgb_image.shape[1], scaled_rgb_image.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.original_label.setPixmap(pixmap)

            # 准备数据张量
            self.data_tensor = torch.tensor(self.data.reshape(-1, len(self.wavelengths)),
                                            dtype=torch.float32).unsqueeze(1)

    def predict_and_show(self):
        if self.data_tensor is not None:
            # 进行预测
            with torch.no_grad():
                predictions = []
                batch_size = 1024
                for i in range(0, self.data_tensor.size(0), batch_size):
                    batch_data = self.data_tensor[i:i + batch_size]
                    batch_preds = self.net(batch_data).argmax(dim=1).cpu().numpy()
                    predictions.append(batch_preds)
                predictions = np.concatenate(predictions)

                # 将预测结果转换回图像大小
                predicted_image = predictions.reshape(self.image_size[1], self.image_size[0])

                # 创建与标签图等大的RGB图像
                output_image = Image.new("RGB", (self.image_size[0], self.image_size[1]))

                # 颜色映射
                color_map = {
                    0: (255, 255, 255),
                    1: (255, 0, 0),
                    2: (0, 255, 0),
                    3: (0, 0, 255),
                    4: (255, 255, 0),
                    5: (128, 0, 128),
                    6: (0, 0, 0)
                }

                # 根据预测结果填充图像
                for i in range(self.image_size[1]):
                    for j in range(self.image_size[0]):
                        pixel_class = predicted_image[i, j]
                        output_image.putpixel((j, i), color_map[pixel_class])

                # 显示预测图像
                # 对预测图像进行缩放
                scaled_output_image = cv2.resize(np.array(output_image), (300, 320),
                                                 interpolation=cv2.INTER_NEAREST)
                qimg = QImage(scaled_output_image, scaled_output_image.shape[1], scaled_output_image.shape[0],
                              QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)
                self.predicted_label.setPixmap(pixmap)

                # 显示保存对话框
                save_dialog = SaveDialog(self.rgb_image, output_image, self)  # 传递self作为父窗口
                if save_dialog.exec_():
                    print("Image saved successfully.")
                else:
                    print("No image saved.")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = AppDemo()
    demo.show()
    sys.exit(app.exec_())