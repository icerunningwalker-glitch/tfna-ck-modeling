# TH21 tFNA 透皮输运模块

## 运行

从统一项目根目录执行：

```text
python src/transport/run_model.py
python src/transport/run_model.py --output-dir results/transport/my_run
python src/transport/verify_results.py --generated-dir results/transport/my_run
python src/transport/verify_results.py --generated-dir results/transport/my_run --report results/transport/my_run/validation.json
```

脚本根据自身位置定位输入，所以也可从任意工作目录使用脚本的完整路径执行。默认新结果保存到 `results/transport/generated/`；入口拒绝将输出写入附带的 `data/`、`models/`、`logs/`、参考结果根目录或参考图像目录。参数固定读取 `data/transport/reference_inputs.json`，15 分钟汇总和图像也按该参考敷用条件生成。更改应用时长或装载比例时，需要同步调整入口的取样、剂量与汇总计算。

本模块需 NumPy、SciPy、pandas 和 Matplotlib；原始依赖版本记录在本目录的 `requirements.txt`。不需要下载模型、联网、训练或运行分子动力学。常规执行需数分钟，取决于计算机配置。

## 内容与输出

- `physics.py`：颗粒物性、皮肤层扩散、毛囊储库及壁通透性换算。
- `model.py`：有限体积离散和 BDF 前向积分。
- `run_model.py`：完整前向模拟、物理校验和输出。
- `figure_style.py`、`plot_depth_heatmap.py`：图形样式与时间—深度图。
- `verify_results.py`：将新运行的数值表与附带参考结果逐项比较。
- `data/transport/`：原有参数、设计输入、拟合摘要和整理后的文献数值。
- `logs/transport/Data_Sources.csv`：文献来源及 DOI。
- `models/transport/MODEL_CARD.md`：用途、假设与限制；本模块没有训练权重。
- `results/transport/`：附带参考结果；生成结果写入其独立子目录。

主要输出是 `summary.json`、`timeseries.csv`、`layer_retention.csv`、`depth_profiles.csv`、`depth_time_heatmap.csv`、`uptake_sensitivity.csv` 和图像。CSV 表头标明时间、深度、质量及浓度单位。运行检查质量守恒、状态非负性，以及敷料移除后停止外部补给。

`summary.json` 的历史字段 `dose_ug` 按总输入核心 DNA 当量（ng/cm²）除以 1000 得到，实际单位为 **µg/cm²**；对应 1 cm² 模拟面积时，其数值也等于该面积内的总 µg。该字段名和数值沿用参考结果。`run_model.py` 输出完整载体的质量和浓度，没有独立的游离 CK 或 CK 装载当量输出列。若采用团队设定的每个 tFNA 装载 2 个 CK，可将完整载体摩尔浓度乘以 2，另行报告为 CK 装载当量；这是按给定装载比例进行的算术换算。

复核入口对 5 张非空数值表逐项比较，要求列顺序、行数和文本字段一致、所有数值有限，数值容差为 `rtol=1e-9`、`atol=1e-10`；同时比较汇总值和实际运行输入，并检查物理残差阈值。`summary.json` 中的 3 个残差不要求与历史值相等，而要求质量相对残差 `<1e-7`、最小状态 `>-1e-7`、移除后新增输入变化 `<1e-8`。

本次合并后已从独立工作目录完整执行前向模拟，执行时间为 10.64 秒；5 张数值表与附带结果在复核容差内一致。输入及附带结果共 31 个文件的内容保持不变。校验记录见 `logs/transport/validation.json` 和 `logs/transport/provenance_and_run.json`。运行时间仅代表本次环境，不保证其他设备相同。

## 证据边界

本入口复现参考 TH21 完整载体的前向模拟。团队设计条件为每个 tFNA 装载 2 个 CK，按此比例，1 µmol/L CK 装载当量对应 0.5 µmol/L tFNA。参数拟合摘要和文献记录保留在 `data/transport/` 与 `logs/transport/`，入口使用现有拟合值执行前向计算。
