# 第三方来源与适配记录

- Ultralytics YOLOv8：AGPL-3.0，固定提交`44c7c3514d87a5e05cfb14dba5a3eeb6eb860e70`，源码版本8.0.90。本项目的训练子类继承其框架，保留上游许可。
- Torch-Pruning：MIT，发行版本1.6.1；YOLOv8的C2f拆分思路参考其作者示例，原示例注明Hyunseok Kim贡献。`pruning.py`保留来源说明。
- NEU-DET：东北大学宋克臣、严云辉等作者提供的公开研究数据。使用时按官方主页引用相关研究；本仓库不重新发布原始数据包。
- COCO预训练YOLOv8n权重：从Ultralytics官方assets发布页下载，不把它说成本课程自行预训练的成果。
- ONNX1.17.0与ONNX Runtime1.20.1：用于静态模型导出、格式检查及CPU运行。依赖通过官方PyPI发行哈希验证，许可证保留在本机局部依赖中；不随轻量代码包重新分发二进制依赖。来源为[ONNX](https://github.com/onnx/onnx/tree/v1.17.0)和[ONNX Runtime](https://github.com/microsoft/onnxruntime/tree/v1.20.1)。
- PyTorch2.5.1：补充诊断参考其公开Conv/BN融合及BN统计更新实现，并使用已固定框架的融合入口。仅更新BN统计的适配、CUDA/CPU计时与ONNX数值验证均在本工程中保存。

主要本地适配：

1. 固定旧版检测接口，在项目内加载源码；记录一行配置目录惰性求值修正，避免未使用的全局目录探测。
2. 将临时文件与配置放在项目目录；关闭可选同步和自动依赖安装。
3. 对NumPy新版本提供`trapz → trapezoid`的进程内兼容别名。
4. 对C2f等价改写检查原始预测一致性，创建可训练参数时避免InferenceMode，并在依赖追踪前重建检测头缓存。
5. 使用保留剪枝模型结构的训练入口和FP32模型保存；明确检查点不包含优化器恢复状态。
6. 使用自建结果绘图与检测案例展示，避开旧版可视化对已移除Pillow接口的依赖。

所有改动均限于本课设工程和下载的源码副本，未修改全局PyTorch、CUDA或已有学习工程。

数据建议引用：

- K. Song and Y. Yan. A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects. Applied Surface Science, 285, 858–864, 2013.
- Yu He, Kechen Song, Qinggang Meng, Yunhui Yan. An End-to-End Steel Surface Defect Detection Approach via Fusing Multiple Hierarchical Features. IEEE Transactions on Instrumentation and Measurement, 69(4), 1493–1504, 2020.
- Gongfan Fang et al. DepGraph: Towards Any Structural Pruning. CVPR, 2023.

引用字段提交前应按原始论文核对；本项目保留官方来源链接，实验结果不跨数据划分与文献直接排名。
