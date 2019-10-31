
## 抽象定义
我们将 一个完整的可以走通的结构称为模型。
包括 backbone + Dataset + loss 等， 包括数据的模块流通，称为Model。在这个抽象层里面将所有的模块进行组合

- 如果没有特别的传播，我们会有一个基础的model. 称为vanilla_model. 这个model可以完成大多数基本的任务
- 如果有特别的任务，比如我写的一个模型包含好几个nn module. 如encoder-decoder这样的结构，GAN的结构。
那么我们能够在model里面将不同的模块组合起来。

module为自定义的更小的模块，比如switch_norm层，自定义的一些层，比如conv_TBC这样的层


最终的设计就是，在model中完成模型的组装. 不特殊的就直接使用vanilla_model完成自动化组装。
特殊的就自己定义model
