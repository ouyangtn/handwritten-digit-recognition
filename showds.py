#该代码用于了解mnist数据集的基本信息，包括数据的形状、特征和标签。
import tensorflow as tf #导入tensorflow和matplotlib库
from matplotlib import pyplot as plt

mnist = tf.keras.datasets.mnist#加载mnist数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()#并将数据集划分为训练集和测试集

# 可视化训练集输入特征的第一个元素
plt.imshow(x_train[0], cmap='gray')  # 绘制灰度图
plt.show()

# 打印出训练集输入特征的第一个元素
print("x_train[0]:\n", x_train[0])
# 打印出训练集标签的第一个元素
print("y_train[0]:\n", y_train[0])

# 打印出整个训练集输入特征形状
print("x_train.shape:\n", x_train.shape)
# 打印出整个训练集标签的形状
print("y_train.shape:\n", y_train.shape)
# 打印出整个测试集输入特征的形状
print("x_test.shape:\n", x_test.shape)
# 打印出整个测试集标签的形状
print("y_test.shape:\n", y_test.shape)
