# 附件5代码提交要求核对

| 要求 | 实际交付 | 状态/待补 |
|---|---|---|
| 源码、依赖、运行命令 | 根目录入口；design/benchmark源码；两级requirements；README | 已提供；全量原环境为Linux，轻量推理可Windows运行 |
| 自训练入口及参数 | 历史MLP最终拟合入口；新版七算法五折入口；参数/分折/选择loss表 | 已提供；本次未重新训练 |
| 预训练模型、版本、调用信息 | 冻结HyenaDNA官方revision/文件hash；历史MLP joblib+NPZ；benchmark折模型/TabPFN上下文 | 已提供；TabPFN V2权重需按官方许可自行获取 |
| 一键最终候选清单 | predict.py → UTF-8 results.csv；已运行示例在results/ | 已提供；官方赛道与专用提交模板未给，默认UNSPECIFIED |
| 设计→筛选示例 | design.py复算已有池并重放筛选；--generate可重新设计 | 已提供并检查已有池重放；未重新生成候选 |
| 数据来源、预处理、划分 | DATA_SOURCES.md、Model Card及分折表 | 已提供；原数据许可文件/完整获取时间待确认 |
| 日志、随机种子、实际验证 | 已脱敏历史结果/参数记录与validation；新输出run.json | 已提供；原训练硬件/驱动/耗时未完整保存 |
| Notebook | 主入口完整覆盖核心流程 | 按附件规则无需额外Notebook |
| 三维结构 | 本包只进行序列优化，结构字段留空 | 不含三维结构筛选；若赛道要求请另附实际PDB/CIF及对应关系 |
| 第三方许可 | 公开来源、revision、许可说明 | 模型卡声明BSD-3-Clause；历史副本没有单独版权许可文件，见notice，不编造抬头；自有代码、文档、对接标签与DNA序列设计采用MIT，第三方模型保留原许可 |
| 隐私 | 分发文本、压缩成员和模型/数组检查；相对路径manifest | 最终实际结果见validation/package_validation.json |

验收记录区分“代码已交付”“本次实际运行”“尚无原始记录”，不将待补信息写成完成事实。此包适合交给他人复用与复核；正式赛事提交仍需核对指定赛道模板与数据/知识产权权限。
