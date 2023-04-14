# CNN Models
- [从LeNet到SENet——卷积神经网络回顾](https://zhuanlan.zhihu.com/p/33845247)
- 1989 Le CUN [經典論文: Backpropagation Applied to Handwritten Zip Code Recognition](https://ieeexplore.ieee.org/document/6795724)
- 1998 LeNet
  - Lecun, Y.; Bottou, L.; Bengio, Y.; Haffner, P. (1998). "Gradient-based learning applied to document recognition" . Proceedings of the IEEE. 86 (11): 2278–2324.
- 2009  ImageNet (since 2009) [經典論文: ImageNet: A large-scale hierarchical image database](https://ieeexplore.ieee.org/document/5206848)
  - ImageNet專案是一個大型視覺資料庫，用於視覺目標辨識軟體研究。
  - 該專案已手動注釋了1400多萬張圖像，以指出圖片中的物件，並在至少100萬張圖像中提供了邊框。
  - ImageNet包含2萬多個典型類別，例如「氣球」或「草莓」，每一類包含數百張圖像。
  - 儘管實際圖像不歸ImageNet所有，但可以直接從ImageNet免費獲得標註的第三方圖像URL。
  - 2010年以來，ImageNet專案每年舉辦一次軟體競賽，即[ImageNet大規模視覺辨識挑戰賽(ILSVRC)](https://www.image-net.org/challenges/LSVRC/)
  - 挑戰賽使用1000個「整理」後的非重疊類，軟體程式比賽正確分類和檢測目標及場景
  - [ImageNet Dataset | Papers With Code](https://paperswithcode.com/dataset/imagenet)
    - [CoCa: Contrastive Captioners are Image-Text Foundation Models(222)](https://arxiv.org/abs/2205.01917v2) 
    - [PyTorch實作](https://github.com/lucidrains/CoCa-pytorch)
- [ImageNet大規模視覺辨識挑戰賽(ILSVRC)ImageNet Large Scale Visual Recognition Challenge](https://www.image-net.org/challenges/LSVRC/)
- [2010 ILSVRC](https://www.image-net.org/challenges/LSVRC/2010/index.php) 使用 Large-scale SVM classification
- [2011 ILSVRC](https://www.image-net.org/challenges/LSVRC/2011/index.php) XRCE |Florent Perronnin, Jorge Sanchez | Compressed Fisher vectors for Large Scale Visual Recognition
- [2012 ILSVRC](https://www.image-net.org/challenges/LSVRC/2012/index.php)
  - AlexNet(2012)::大突破
  - [經典論文:ImageNet Classification with Deep Convolutional Neural Networks (2012)](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
  - Hinton學生Alex Krizhevsky於2012年提出並拿下ILSVRC”12的冠軍讓CNN重返榮耀
  - 其將top-5 error減少至15.3% ， outperform同年度第二名26.2%
  - LeNet5的加強版
  - 主要的新技術與應用::將ReLU, Dropout, LRN加到model中
  - 用GPU來加快training效率
  - data augmentation增加訓練資料集
  - [PyTorch實作是以這篇論文為主One weird trick for parallelizing convolutional neural networks(2014)](https://arxiv.org/abs/1404.5997) 
- [2013 ILSVRC](https://www.image-net.org/challenges/LSVRC/2013/index.php)
- [2014 ILSVRC](https://www.image-net.org/challenges/LSVRC/2014/index.php)
  - 👍[經典論文:VGG model: Very Deep Convolutional Networks for Large-Scale Image Recognition(2014)](https://arxiv.org/abs/1409.1556)
    - 在AlexNet之後，另一個提升很大的網路是VGG，ImageNet上Top5錯誤率減小到7.3%。
    - 主要改進就是：更深！網路層數由AlexNet的8層增至16和19層，更深的網路意味著更強大的網路能力，也意味著需要更強大的計算力，還好，硬體發展也很快，顯卡運算力也在快速增長，助推深度學習的快速發展。
    - 同時只使用3x3的卷積核，因為兩個3x3的感受野相當於一個5x5，同時參數量更少，之後的網路都基本遵循這個範式。
  - Inception v1 model(2014)
    - [經典論文:GoogleNet model:Going Deeper with Convolutions(2014)](https://arxiv.org/abs/1409.4842)
    - ImageNet Top5錯誤率6.7%
    - GoogLeNet從另一個維度來增加網路能力，每單元有許多層平行計算，讓網路更寬了\
    - 通過網路的水準排布，可以用較淺的網路得到很好的模型能力，並進行多特徵融合，同時更容易訓練
    - 另外，為了減少計算量，使用了1x1卷積來先對特徵通道進行降維。
    - 堆疊Inception模組而來就叫Inception網路，而GoogLeNet就是一個精心設計的性能良好的Inception網路（Inception v1）的實例。
    - 但是，網路太深無法很好訓練的問題還是沒有解決  == >  ResNet提出residual connection
- [2015 ILSVRC](https://www.image-net.org/challenges/LSVRC/2015/index.php)
  - 第一名:[經典論文:ResNet model: Deep Residual Learning for Image Recognition(2015)](https://arxiv.org/abs/1512.03385)
    - 引入 `殘差網路` Residual Connections
    - 可以容易地訓練避免梯度消失的問題，所以可以得到很深的網路，網路層數由GoogLeNet的22層到了ResNet的152層
  - Inception v2（BN-Inception）：2015，Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
    - ImageNet Top5错误率：4.8%
  - Inception V3 model(2015)
    - [經典論文:Google InceptionV3 model:Rethinking the Inception Architecture for Computer Vision(2015)](https://arxiv.org/abs/1512.00567)
- [2016 ILSVRC](https://www.image-net.org/challenges/LSVRC/2016/index.php)
  - 第一名:[ResNext model:Aggregated Residual Transformations for Deep Neural Networks(2016)](https://arxiv.org/abs/1611.05431v2)
  - Inception V4 model(2016)
    - [經典論文: Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning(2016)](https://arxiv.org/abs/1602.07261)
  - [DenseNet model: Densely Connected Convolutional Networks(2016)](https://arxiv.org/abs/1608.06993)
- [2017 ILSVRC](https://www.image-net.org/challenges/LSVRC/2017/index.php)
  - 第一名:[SqueezeNet model: SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size(2016)](https://arxiv.org/abs/1602.07360)
- The end 
 
- [ShuffleNet V2 model: ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design(2018)](https://arxiv.org/abs/1807.11164)
- [EfficientNetV2 model: EfficientNetV2: Smaller Models and Faster Training(2021)](https://arxiv.org/abs/2104.00298)  [GITHUB](https://github.com/google/automl/tree/master/efficientnetv2)

# REVIEW
- [A Survey of the Recent Architectures of Deep Convolutional Neural Networks(2019)](https://arxiv.org/abs/1901.06032)

# 如何使用別人的模型 ==> transfer learning
- 使用[tf.keras.applications](https://www.tensorflow.org/hub/)內建的
- 使用別人在Tensorflow HUB分享的

## https://www.tensorflow.org/api_docs/python/tf/keras/applications
```
densenet module: DenseNet models for Keras.

imagenet_utils module: Utilities for ImageNet data preprocessing & prediction decoding.

inception_resnet_v2 module: Inception-ResNet V2 model for Keras.
inception_v3 module: Inception V3 model for Keras.

mobilenet module: MobileNet v1 models for Keras.
mobilenet_v2 module: MobileNet v2 models for Keras.

nasnet module: NASNet-A models for Keras.

resnet module: ResNet models for Keras.
resnet50 module: Public API for tf.keras.applications.resnet50 namespace.
resnet_v2 module: ResNet v2 models for Keras.

vgg16 module: VGG16 model for Keras.
vgg19 module: VGG19 model for Keras.

xception module: Xception V1 model for Keras.
```

```python
from tensorflow.keras.applications import Xception
xception = Xception()
xception.summary()
```
