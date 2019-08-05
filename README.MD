# Introduction
Universal pytorch train & test framework, 
which is developed for fast implementation, research and deploy 

# Getting Started

实现参照
- pytorch cycle-gan
- fairseq
- mmdetection

# TODO
- [X] 基础组件的拆分: backbone, dataset, loss, post_process
- [X] 新的全局组件注册机制，全局文件配置
- [ ] Bug fixes: 注册的函数类型后处理,支持部分参数给定
- [ ] Bug fixes: 支持配置pytorch内置的loss
- [ ] 通用的流程 pipe_line抽象, 添加服务代码
- [ ] visdom训练loss可视化，以及结果的可视化
- [ ] 支持训练时评估：从dataset 和 临时的文件夹。 需要可配置是否使用
- [ ] 提供lmdb, HDF5数据库的通用读写工具
- [ ] 提供已经deploy ready 的模型的Tensor RT转换工具, 模板代码
- [ ] 高效的多卡分布式训练

易用性
- [ ] 使用文档及范例
- [ ] 提供更多的基础组件的实现， 比如efficient Net, resnet FPN 等， 
常用的loss, 如dice loss, centre loss, arc face等


# This project is still under development
Now including 2 text detection methods
- C^2TD: a fast and robust text detector via centre line regression and centre border probability
- CRNN
 
