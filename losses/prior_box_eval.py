import torch
from itertools import product as product
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


class PriorBoxEval(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    Note:
    This 'layer' has changed between versions of the original SSD
    paper, so we include both versions, but note v2 is the most tested and most
    recent version of the paper.

    """
    def __init__(self, cfg):
        super(PriorBoxEval, self).__init__()
        self.image_size = cfg['min_dim']  # 是先h, 再w
        self.variance = cfg['variance'] or [0.1]  # 并未用到
        self.feature_maps = cfg['feature_maps']

        self.defaultbox = cfg['defaultbox']

        self.steps = cfg['steps']
        self.clip = cfg['clip']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, (map_h, map_w) in enumerate(self.feature_maps):
            for i, j in product(range(map_h), range(map_w)):  # 对feature map的每一个点
                # 存储的是相对于整幅图的大小， cx,cy是中心位置，s_K是长款，都是相对于原图的相对坐标，s_K应该是不同的
                f_k_h = self.image_size[0] / self.steps[k]
                f_k_w = self.image_size[1] / self.steps[k]
                cx = (j + 0.5) / f_k_w
                cy = (i + 0.5) / f_k_h

                # 针对预置的5个default box生成相关的
                for box in self.defaultbox:
                    s_k_x = box[0]/self.image_size[1]
                    s_k_y = box[1]/self.image_size[0]
                    mean += [cx, cy, s_k_x, s_k_y]

        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
