'''图像识别程序。通读取指定路径的图像文件，对图像进行预处理（包括调整大小、二值化、归一化），
然后用加载的模型进行预测试，输出预测的数字
'''
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import re

def check(self, image_path):#
    searchstr = "_"#检查下划线
    res = re.search(searchstr, image_path)
    if res != None:
        return False
    else:
        return True
#定义了模型的保存路径和模型结构
model_save_path = r'D:\Pyhton_exp\handwritten-digit-recognition\model\nummodel.ckpt'
#用tf.keras创建了一个Sequential模型，包含Flatten层、Dense层和Softmax层。
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

#加载模型参数
status=model.load_weights(model_save_path)
#根据参数初始化指定的变量
status.expect_partial()#等待并匹配部分输出。当控制台输出中出现与匹配模式部分匹配的内容时，函数将返回匹配到的内容，并继续等待完整的输出
preNum = 1
image_path="./png/7-7.png"
for i in range(preNum):
    #读取图片
    img = Image.open(image_path)
    #读取图片，并将图片转换为Numpy数组
    image = plt.imread(image_path)

    #处理图片，将图片大小调整为28×28并进行二值化处理
    img = img.resize((28, 28), Image.ANTIALIAS)

    # 将PIL库中的图片从彩色转换为灰度图像，再将PIL图像转换为Numpy数组
    img_arr = np.array(img.convert('L'))
    #展示图片
    plt.set_cmap('gray')
    plt.imshow(img_arr)
    # 由于训练集是黑底白字，所以转图片转换为黑底白字
    # 颜色取反：img_arr=255-img_arr
    # 让输入图片变成只有黑色和白色的高对比度图
    if check(image_path):
        for i in range(28):
            for j in range(28):
                if img_arr[i][j] < 200:#，将像素值小于200的像素设为纯白色（255），否则设为纯黑色（0）。
                    # 纯白色
                    img_arr[i][j] = 255
                else:
                    # 纯黑色
                    img_arr[i][j] = 0
    # 归一化
    img_arr = img_arr / 255.0
    #将Numpy数组转换成TensorFlow张量，同时在第0维增加一个新的维度，让img_arr变成一个批次为1的图像样本，可以直接传入神经网络模型进行预测
    x_predict = cimg_arr[tf.newaxis, ...]
    #使用模型进行预测
    result = model.predict(x_predict)
    #第一个维度表示批次中的样本数（0-9），第二个维度表示每个样本的分类数（0-9的概率）
    # 返回的pre是张量，获取预测结果中最大值所在位置的下标
    pred = tf.argmax(result, axis=1)

    print("图片", image_path, "的数字为:", end="")

    # 将预测结果张量转换为numpy数组
    x = pred.numpy()
    print(x[0])
    plt.pause(1)
    plt.close()
