## 模型相关的参数

设计的时候，模型不分所谓的 头,neck, backbone.

特殊的结构，比如RFB的head？ 

我们还没有到这样细分的程度

比如一个网络：
VGG  有几个创建VGG的函数。（这几个函数应该是private，不允许导出）
在VGG中可以有一个模型， 比如VGG的FPN,  基于VGG的一个检测模型
这两个模型都继承于nn.Moudle. 使用装饰器的模式去注册这个模型


## 抽象定义
我们将 一个完整的可以走通的结构称为模型。
包括 backbone + Dataset + loss 等， 包括数据的模块流通，称为Model。在这个抽象层里面将所有的模块进行组合

1) 如果没有特别的传播，我们会有一个基础的model. 称为vanilla_model. 这个model可以完成大多数基本的任务
2) 如果有特别的任务，比如我写的一个模型包含好几个nn module. 如encoder-decoder这样的结构，GAN的结构。
那么我们能够在model里面将不同的模块组合起来。

module为自定义的更小的模块，比如switch_norm层，自定义的一些层，比如conv_TBC这样的层


---- 注册到哪一个层级，要不我还是只注册到model这一个层级吧， 不同的模型参数不同呐。
如果你想要只传入一个arg， 那么你自己的模型要变动.
mmdetection 方法的做法是


