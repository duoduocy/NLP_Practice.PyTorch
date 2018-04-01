## 基于LSTM的古诗词生成

## 诗词来源
首先你需要下载数据：
https://github.com/jackeyGao/chinese-poetry
总的训练样本大概有50k首诗。

## 训练
注意！默认是用GPU的
```
python train.py
```

## 测试
```bash
python sample.py
```

## 参考
1. Paper [Chinese Poetry Generation with Recurrent Neural Networks](http://www.aclweb.org/anthology/D14-1074)
2. Codes: [ch-poetry-nlg](https://github.com/ne7ermore/torch_light/tree/master/ch-poetry-nlg) and [pytorch-poetry-gen](https://github.com/justdark/pytorch-poetry-gen/)
