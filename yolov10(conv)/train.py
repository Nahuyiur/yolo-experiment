from ultralytics import YOLOv10
import warnings
warnings.filterwarnings('ignore')
# 模型配置文件
model_yaml_path = r"/root/lanyun-tmp/yolov10/ultralytics/cfg/models/v10/yolov10n.yaml"
#数据集配置文件
data_yaml_path = r'/root/lanyun-tmp/yolov10/data/data.yaml'
#预训练模型
#pre_model_name = 'yolov10n.pt'
if __name__ == '__main__':
    # #加载预训练模型
    # model = YOLOv10(model_yaml_path).load(pre_model_name)
    # 不加载预训练模型
    model = YOLOv10(model_yaml_path)
    #训练模型
    results = model.train(data=data_yaml_path,
                          imgsz=640,
                          epochs=300,
                          batch=16,
                          workers=8,
                          optimizer='SGD',  # using SGD
                          amp=False,  # 如果出现训练损失为Nan可以关闭amp
                          project='runs/V10train',
                          name='exp',
                          )