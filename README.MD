# 文本检测识别工具包
Jesse Wong at aiyunxiao
- 封装多个检测算法， 保证多个算法解耦可配置添加。“反射”方式解耦
- 能够快速实验。 任何可调参数，一定要通过option添加。 （这样做有很大的好处， 或许以开始觉得，复制一份代码，创建一个分支，再改改，更快速，
但后期这样很难维护）
- 算法的模块，网络结构 能够模块化。方便快速新算法尝试.


# 基本思路：
采用类似于“反射”的方式来保证
- 网络模型可配置， 而不用修改任何代码
- 装载数据的dataset可配置
- 模型的基本功能抽象，如保存、装载、打印损失等功能

实现参照pytorch cycle-gan 项目

# This project is still under development
Now including 2 text detection methods
- C^2TD: a fast and robust text detector via centre line regression and centre border probability
- anchor-based key point detection 
 


