# Introduction
Universal pytorch train & test framework, 
which is developed for fast implementation, research and deploy 

# Getting Started

基础设计实现参照, Thanks for their brilliant works 
- pytorch cycle-gan
- fairseq
- mmdetection

# TODO LIST
- [X] 基础组件的拆分: backbone, dataset, loss, post_process
- [X] 通用的model组装文件、推理model. (是否需要将推理model集成)
- [X] 组件注册机制，全局文件配置
- [X] 配置的方式实现loss加权、打印
- [X] 模型初始化参数统一配置:normal, xavier, kaiming, orthogonal. !!支持按照不同的模块初始化
- [X] 预处理可配置pipe_line: 图像-常见的pad, resize, toTensor(到0~1)，通道转换
- [ ] deep NLP 与 图像 有很大的不同。利用torchtext实现一个针对NLP的通用预处理
- [ ] feature: 注册的函数类型后处理,支持部分参数给定. 例如，使用functools.partial
- [ ] feature: 支持配置pytorch内置的loss
- [ ] feature； weight-decay 配置
- [ ] ~~feature: backbone支持函数返回model~~
- [ ] 通用的流程 pipe_line抽象, 如：检测+识别，添加服务代码
- [X] visdom训练loss可视化，以及结果的可视化. 代码中启动visdom, 不能手动启动visdom
- [X] 支持训练时评估：从dataset 和 临时的文件夹。 需要可配置是否使用
- [ ] 提供lmdb, HDF5数据库的通用读写工具、模板代码
- [ ] 提供已经deploy ready 的模型的Tensor RT转换工具, 模板代码
- [ ] DataParallel之外的，高效的多卡分布式训练。pytorch自带的及第三方的

易用性
- [ ] 使用文档及范例
- [ ] 提供更多的常用基础组件的实现， 比如efficient Net, resnet FPN 等， 
常用的loss, 如dice loss, centre loss, arc face等, anchor方法等一系列相关东西


# This project is still under development.
The interface is unstable.

Now including:
- C^2TD: a fast and robust text detector via centre line regression and centre border probability[.config\c2td.py]
- CRNN & GRCNN: only inference
- pixel-link: Codes are from https://git.iyunxiao.com/DeepVision/pytorch-ocr-framework
- im2latex: 少量没有解耦. Codes are from https://git.iyunxiao.com/DeepVision/pytorch-ocr-framework
- Unet segmentation
- To be continue: Transformers

# 如何贡献：

（欢迎使用并回馈代码，使其变得更好）

- 请按照统一的代码风格，如pycharm默认的检查风格

- 不允许随意添加分支。若添加，新增加的分支应该考虑如何能够与原master分支合并，并保证之前的代码兼容。应当持续思考如何使项目变得通用、方便使用
（为了避免使的代码变得更乱）

- 明确贡献和责任。记录你的改动，在新增加文件的开始署名，或修改历史署名，更新历史。

# 更新历史
Jesse Wong
- 初步的框架设计，统一的训练、推理抽象。完成1-5点

hcn
- 增加文字识别模型resnet的backbone添加进来，并将beamsearch后处理及文字后处理整合到ctc_decode.py后处理，
但是这部分代码一直在变动，后处理的代码应该再考虑怎么更好加进来
- ~~单例推理预处理部分，这里是参考了注册机制，每一个模型的预处理注册一个预处理函数，但是感觉这样不太好，
很多预处理的代码有重复部分，可能要考虑怎么抽取出来，用参数配置的方式~~
- 增加im2latext及pixellink训练代码

Jesse Wong
- visdom子进程启动。可视化显示loss. 等待"支持训练时评估"功能完成后，接入更多的可视化展示


hzc
- 增加OCR通用识别模型， 检测，识别dataset
- 训练预处理可配置pipe_line (常用预处理操作)
- 增加evaluators模块供训练时评估