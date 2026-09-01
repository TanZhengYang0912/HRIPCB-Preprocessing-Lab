# Bilateral + AGCWD：成熟项目如何选择组合

## 结论先行

成熟的视觉框架并不会因为两个模块各自的单项分数最高，就直接把它们串起来。它们把串联后的流程当成一个新的候选 pipeline：固定数据、模型和评估协议，做基线、单项、不同顺序和组合的消融，再按任务指标和工程约束选出候选。原因是组合会改变输入分布，而且两个变换的交互可能是非线性的；“单项最佳”不能推出“组合最佳”。这是根据 Albumentations 的组合/消融指导和下列检测框架的 pipeline 设计得出的实验结论。

## 成熟项目的共同做法

### 1. 训练增强与验证/推理预处理分开

- [Ultralytics 数据集代码](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/dataset.py#L304-L333)在 `augment=True` 时构造训练增强；否则只走确定性的 `LetterBox`/格式化流程。[官方默认配置](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml#L63-L70)也把 test-time augmentation (`augment`) 默认设为 `False`。
- [MMDetection 的 COCO 配置](https://github.com/open-mmlab/mmdetection/blob/main/configs/_base_/datasets/coco_detection.py#L20-L73)明确写出 `train_pipeline`（如 `RandomResize`、`RandomFlip`）与 `test_pipeline`（确定性的 `Resize`、无随机翻转），并分别绑定 train/val/test dataloader；val/test 还关闭 `shuffle`。
- [Albumentations 官方增强指南](https://albumentations.ai/docs/1-introduction/what-are-image-augmentations/#534-543)明确建议验证集和测试集不使用随机增强；TTA 是独立的推理策略，且会增加延迟（同页 [#516-526](https://albumentations.ai/docs/1-introduction/what-are-image-augmentations/#516-526)）。

因此，Bilateral + AGCWD 若是部署前的确定性图像处理，应在训练、验证、测试和部署中保持同一顺序与参数；若只是比较“冻结模型对输入变换的敏感性”，可以用同一 checkpoint 做验证消融，但最终模型仍要用选中的完整训练协议重新训练并在 test 上只评估一次。

### 2. 先有无增强基线，再做一次只改变一个因素的消融

[Albumentations 官方指南](https://albumentations.ai/docs/1-introduction/what-are-image-augmentations/#424-472)给出的流程是：保留 no-augmentation baseline，先用保守强度，逐个改变一个轴（类型、概率或幅度），记录总体、每类和重要子集的指标；如果叠加导致退化，优先降低幅度或概率，或移除该变换。指南还提醒，多种增强叠加会产生 destructive interaction：每个增强单独有效，叠加后仍可能破坏小目标/细节。

Albumentations 的 [Compose 实现](https://github.com/albumentations-team/albumentations/blob/main/albumentations/core/composition.py#L830-L867)按列表顺序逐个执行变换；[OneOf 实现](https://github.com/albumentations-team/albumentations/blob/main/albumentations/core/composition.py#L1327-L1342)则从候选中选择一个，而不是默认全部串联。这说明成熟库提供组合机制，但不替用户宣称某种组合一定更好；选择仍靠受控实验。[概率文档](https://albumentations.ai/docs/2-core-concepts/probabilities/#128-185)也建议显式记录每个变换的 `p`，因为嵌套概率会改变实际施加概率。

对本项目最小的组合消融应包含：

1. 原图 baseline；
2. Bilateral only；
3. AGCWD only；
4. Bilateral → AGCWD；
5. AGCWD → Bilateral（顺序本身也是变量）。

组合候选不能只复用两个单项冠军的参数。至少要在组合流程中重新扫 Bilateral 的 `diameter/sigmaColor/sigmaSpace` 与 AGCWD 的强度；如果串联损失小焊盘、线条或缺陷边缘，先减小强度、降低应用概率或只保留一个模块。对确定性推理流程，“概率”通常应固定为 1；概率/随机增强只适用于训练增强实验。

### 3. 参数是受控搜索，不是凭视觉挑一个

- [OpenCV bilateralFilter 文档](https://docs.opencv.org/4.13.0/d4/d86/group__imgproc__filter.html)说明：较小 `sigmaColor` 影响很弱，较大值会把更远的颜色混合并产生更强、甚至卡通化的平滑；`d=5`适合实时，`d=9`适合离线，`d>5`明显更慢。它给出参数含义和速度/效果取舍，但没有针对目标检测的“最佳组合”默认值。
- [Ultralytics 官方超参数调优指南](https://github.com/ultralytics/ultralytics/blob/main/docs/en/guides/hyperparameter-tuning.md#L204-L239)使用指定搜索空间和预算，通过 trial 的 task fitness 排名候选；这类搜索的关键是先固定搜索空间、预算和评估协议。不要把搜索空间扩大到未经验证的极端增强。
- [Ultralytics 验证文档](https://github.com/ultralytics/ultralytics/blob/main/docs/en/modes/val.md#L215-L258)暴露 `mAP50`、`mAP75`、`mAP50-95`、每类 AP 以及每图像 precision/recall/F1 等结果；[当前 metrics 源码](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L902-L1010)也定义了这些指标。fitness 的具体权重可能随版本变化，所以应锁定版本并同时报告 `mAP50` 与 `mAP50-95`，不要假设所有 YOLO 版本采用同一个权重。
- [MMDetection 的 COCO metric 源码](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/evaluation/metrics/coco_metric.py#L23-L43)使用 COCO AP/AR；默认 IoU 阈值来自 [#L103-L107](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/evaluation/metrics/coco_metric.py#L103-L107) 的 0.50–0.95 序列。因此目标检测组合应以 IoU 区间上的 AP 为主，而不是只看宽松的 mAP50。

## AGCWD 原论文能说明什么

原始论文是 Huang、Cheng、Chiu 的 [*Efficient Contrast Enhancement Using Adaptive Gamma Correction With Weighting Distribution*](https://doi.org/10.1109/TIP.2012.2226047)（IEEE TIP, 2013；[可访问 PDF](https://dept-info.labri.fr/~achille/ti/Activites/Note-de-lecture/Articles/efficient_contrast_enhancement_method_using_adaptative_gamma_correction_with_weighting_distribution_2013.pdf)）。它的核心是从亮度直方图的加权 PDF/CDF 自适应地产生 gamma；权重参数（论文中的 `alpha`）控制分布调整，并在彩色图像的 HSV `V` 通道上处理以尽量保持色相/饱和度。论文主要用视觉效果、亮度误差（AMBE）、颜色失真（ΔE94）、结构/质量指标（FSIM）和速度来评价增强效果；它没有证明 AGCWD 单独或与 Bilateral 串联会提高目标检测 mAP。

这一区分很重要：本仓库 `src/hripcb_preprocessing/filters.py` 中的 [`apply_agcwd`](../../src/hripcb_preprocessing/filters.py#L40-L63)是“AGCWD-style”实现，`gamma` 是自定义的全局强度乘数，不应直接称为原论文的 `alpha`。因此当前 `0.8/1.0/1.2` 是本项目的强度 sweep，而不是论文已经验证过的最佳参数。AMBE、ΔE94、FSIM 可以作为图像质量诊断，但最终 detector winner 必须由检测指标决定。

## 对 Bilateral + AGCWD 的验收建议

建议把 `val` 作为唯一参数选择集、把 `test` 留作最终一次性报告，并保持当前项目的固定模型、图像尺寸、阈值和 seed。候选只有在以下条件都满足时才替换 baseline：

- `mAP50-95` 是主门槛；`mAP50`、precision、recall、F1 作为辅助，不接受只提升 mAP50 但降低主指标的候选；
- 关键缺陷类别、目标尺寸分组和困难子集没有不可接受的回退；
- Bilateral 的额外耗时/吞吐下降在部署预算内（OpenCV 明确提示 bilateral 较慢）；
- 最终候选在重复 seed 或重复训练中仍稳定。Albumentations 指南对最终策略候选建议至少用两个 seed 检查方差；资源有限时，至少对最接近 baseline 的候选做复核；
- 若组合没有超过 baseline，就记录“组合不适合当前 detector/数据分布”，不要因为两个单项各自领先而强行保留。

如果 `Bilateral → AGCWD` 比单项差，下一步应先做减弱强度/减少应用概率和反向顺序的消融；这比继续扩大任意参数网格更符合成熟项目的选择逻辑。

## 主要来源

- [Albumentations：What are image augmentations?](https://albumentations.ai/docs/1-introduction/what-are-image-augmentations/)
- [Albumentations：Pipelines](https://albumentations.ai/docs/2-core-concepts/pipelines/)
- [OpenCV：bilateralFilter](https://docs.opencv.org/4.13.0/d4/d86/group__imgproc__filter.html)
- [Ultralytics：dataset train/eval transforms](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/dataset.py)
- [Ultralytics：validation](https://github.com/ultralytics/ultralytics/blob/main/docs/en/modes/val.md)
- [MMDetection：COCO train/test pipelines](https://github.com/open-mmlab/mmdetection/blob/main/configs/_base_/datasets/coco_detection.py)
- [AGCWD 原论文 DOI](https://doi.org/10.1109/TIP.2012.2226047)
