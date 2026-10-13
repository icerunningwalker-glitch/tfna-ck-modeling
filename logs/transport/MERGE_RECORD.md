# 透皮模块合并记录

- 输入来源：透皮代码包 `final_data_and_src`。
- 5 个 Python 源文件归入 `src/transport/`。
- 原有参数和整理后的数值数据归入 `data/transport/`。
- 原有数值结果及图归入 `results/transport/`，保持原内容。
- 模型说明归入 `models/transport/`，文献出处归入 `logs/transport/`。
- 仅调整模块输入定位、输出目录和说明文件引用；数学方程、参数值、求解器设置及图像内容均保留。
- 默认输出使用 `results/transport/generated/`，防止覆盖附带参考结果。
- 原包仅提供前向模拟入口；本模块未新增参数拟合、AI 候选性能推断或实验结果。
