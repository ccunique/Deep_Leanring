# 对比tensorflow和keras搭建CNN
注：mnist_cnn_tensorflow & mnist_cnn_keras
## 1.tensorflow和keras相同点
二者都是“符号式”的库，都是**先搭建计算图，定义好网络结构，然后再把具体数据feed到模型里面运算**
  
## 2.搭建模型的复杂性
- keras是更高级的库，搭建网络十分简单，下一层直接在上一层的基础上add即可，代码简洁易懂；
- tensorflow灵活性更高，写起来也更复杂，搭建网络需要考虑层与层之间的具体细节，包括上层输出与下层输入的维度匹配等细节都要考虑在内，很容易写错。
