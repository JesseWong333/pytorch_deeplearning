# Introduction
Universal pytorch train & test framework, 
which is developed for fast implementation, research and deploy 

# Getting Started

实现参照
- pytorch cycle-gan
- fairseq
- mmdetection

# TODO LIST
- [X] 基础组件的拆分: backbone, dataset, loss, post_process
- [X] 通用的model组装文件、推理model. (是否需要将推理model集成)
- [X] 组件注册机制，全局文件配置
- [X] 配置的方式实现loss加权、打印
- [X] 模型初始化参数统一配置:normal, xavier, kaiming, orthogonal. !!支持按照不同的模块初始化
- [ ] 预处理可配置pipe_line: 图像-常见的pad, resize, toTensor(到0~1)，通道转换
- [ ] deep NLP 与 图像 有很大的不同。利用torchtext实现一个针对NLP的通用预处理
- [ ] feature: 注册的函数类型后处理,支持部分参数给定. 例如，使用functools.partial
- [ ] feature: 支持配置pytorch内置的loss
- [ ] feature； weight-decay 配置
- [ ] ~~feature: backbone支持函数返回model~~
- [ ] 通用的流程 pipe_line抽象, 如：检测+识别，添加服务代码
- [ ] visdom训练loss可视化，以及结果的可视化. 代码中启动visdom, 不能手动启动visdom
- [ ] 支持训练时评估：从dataset 和 临时的文件夹。 需要可配置是否使用
- [ ] 提供lmdb, HDF5数据库的通用读写工具、模板代码
- [ ] 提供已经deploy ready 的模型的Tensor RT转换工具, 模板代码
- [ ] DataParallel之外的，高效的多卡分布式训练。pytorch自带的及第三方的

易用性
- [ ] 使用文档及范例
- [ ] 提供更多的常用基础组件的实现， 比如efficient Net, resnet FPN 等， 
常用的loss, 如dice loss, centre loss, arc face等, anchor方法等一系列相关东西


# This project is still under development.
The interface is not stable.

Now including 2 methods
- C^2TD: a fast and robust text detector via centre line regression and centre border probability
- CRNN: only inference
- To be continue: Transformers
