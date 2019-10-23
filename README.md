# Capsule

动态路由算法来自：https://kexue.fm/archives/5112

该版本的动态路由略微不同于Hinton原版，在“单数字训练、双数字测试”的准确率上有95%左右。

其他：

1、相比之前实现的版本：https://github.com/XifengGuo/CapsNet-Keras ，我的版本是纯Keras实现的(原来是半Keras半tensorflow)；

2、通过K.local_conv1d函数替代了K.map_fn提升了好几倍的速度，这是因为K.map_fn并不会自动并行，要并行的话需要想办法整合到一个矩阵运算；

3、其次我通过K.conv1d实现了共享参数版的；

4、代码运行环境是Python2.7 + tensorflow 1.8 + keras 2.1.4

## 交流
QQ交流群：67729435，微信群请加机器人微信号spaces_ac_cn
