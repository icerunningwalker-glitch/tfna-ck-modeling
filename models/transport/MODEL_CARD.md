# TH21 tFNA 皮肤输运模型说明

## 用途

估计 15 分钟敷用 TH21 tFNA 后，完整载体在毛囊、活性表皮和浅层真皮中的时空分布、细胞接触浓度及留存。该模型用于解释和比较递送条件，不替代浓度、安全性或药效实验。

## 方法与输入

模型结合有限毛囊储库、毛囊壁输运、组织扩散、细胞摄取和完整结构损失。核心计算由 `src/transport/model.py` 与 `src/transport/physics.py` 实现，参数从 `data/transport/reference_inputs.json` 读取。设计条件见 `data/transport/Team_Design_Input.json`；文献、拟合及参数依据见 `logs/transport/Data_Sources.csv`、`data/transport/Parameter_Table.csv` 和 `data/transport/fit_summary.json`。

这是机理数值模型，没有训练权重。拟合参数用于确定毛囊腔输运阻力、毛囊壁通透性和细胞摄取速率；前向模拟的主入口为 `src/transport/run_model.py`。

## 输出与校验

主入口生成时间序列、不同层的留存量、深度分布、时空浓度表、摄取敏感性表、汇总 JSON 及图像。运行时检查质量守恒、非负状态和敷料移除后停止外部补给。使用 `src/transport/README.md` 中的命令即可重新生成前向模拟结果。

## 适用范围与局限

- 研究对象为完整 TH21 tFNA，质量按 252 nt 核心 DNA 当量报告。
- 15 分钟内采用理想补给边界；移除敷料后无外部补给。
- 未显式模拟角质层直接通路、配方释放过程或完整载体从细胞内再输出。
- 模型给出完整载体浓度和 CK 装载当量，不能据此直接认定游离 CK 的实测浓度或临床安全性。
- 文献数据拟合摘要和整理后的输入数据保留在 `data/transport/`；本包的运行入口复现前向模拟，不重新执行参数拟合。
