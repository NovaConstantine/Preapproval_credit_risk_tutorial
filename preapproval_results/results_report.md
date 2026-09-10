# v6：审批、定价之前的信用风险预测

## 已完成的修改

- 新课程文件：`../Tutorial_revised_v6.ipynb`。完整运行通过，保留执行输出和图像。
- X 使用明确的允许名单，排除 `int_rate` 和 `grade`；`default` 仅作标签。
- 根据使用者确认，`loan_amnt` 表示申请金额。重新计算 `loan_percent_income = loan_amnt / annual_inc`，不使用 CSV 中原有的比例值。
- LR、RF 的原有模型参数不变，在相同的新输入上重新训练。XGBoost 搜索 48 组参数，每组完成 5 折训练集交叉验证。
- 编码、缺失值处理、LR 标准化都只在训练数据上拟合；CV 中每折分别拟合。
- 同样的允许输入记录放在同一组，外层约 70/30 切分及内层 CV 都不跨组。
- 旧版在测试集上选择成本阈值的代码改为训练集 OOF 预测选择。主比较仍使用三个模型统一的 0.50 阈值。
- 业务示例改为审批前风险筛查，不再直接将带类别权重的分数用作定价或预期损失计算中的 PD。

## 变量与时点

| 变量 | 使用方式 | 审批前含义与假设 |
|---|---|---|
| `person_age` | 保留 | 申请时年龄 |
| `annual_inc` | 保留 | 申请时已取得的年收入信息 |
| `home_ownership` | 保留并编码 | 申请时住房状态 |
| `emp_length` | 保留 | 申请时工作年限 |
| `purpose` | 保留并编码 | 申请人填写的贷款用途 |
| `prior_default` | 保留 | 申请时信用档案中已有的历史违约记录；不是本笔贷款的违约结果 |
| `credit_history` | 保留 | 截至申请时的信用历史年限 |
| `loan_amnt` | 保留 | 使用者确认是申请金额，而非审批后的最终金额 |
| `loan_percent_income` | 重新计算 | 申请金额 / 年收入；不是 DTI，不含已有债务的还款负担 |
| `int_rate` | 排除 | 最终定价结果不属于定价前信息 |
| `grade` | 排除 | 机构评级不属于本示例的审批前原始申请信息 |
| `default` | 只作 y | 后续违约标签，不进入 X |

X 共 9 个变量，其中 8 个记录字段、1 个重新计算的比例。住房和用途采用 drop-first one-hot 编码后，实际模型矩阵为 **15 列**。完整列名见 `encoded_feature_names.json`。历史违约和信用历史字段的申请时点含义属于教学假设：CSV 没有快照时间可供独立核实。这里没有 FICO，也没有真正的 DTI。

## 训练与选择方法

数据共 32,572 行。`GroupShuffleSplit(test_size=0.30, random_state=42)` 得到训练集 22,795 行、测试集 9,777 行。分组依据为允许输入 X 的完全相同记录，不使用标签、利率或评级。原来的按行切分在新 X 上有 71 个重复输入组跨越训练和测试，因此 v6 更换了切分方法。

训练集内使用 `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`。所有 48 组 XGBoost 参数都完成 5 折，选择平均验证 AUC 最高的一组，平均 AUC 为 **0.904673**。这是本次搜索中最好的已完成候选，不是全局最优的保证。

选定模型之后，另外使用其训练集 OOF 分数选择筛查阈值，目标为最小化 `max(FP_XGB / FP_RF, FN_XGB / FN_RF)`，其中 RF 使用 0.50 阈值。候选阈值从 0.05 到 0.95、步长 0.005，得到 **0.455**。模型参数和此阈值先写入 `selected_params.json`，再评估测试集。

课程中的非对称成本演示使用另一个目标：假设 FP 成本为 500、FN 成本为 5,000 个教学成本单位。使用训练集 OOF 分数得到成本阈值 **0.190**。该目标愿意接受更多 FP 来减少 FN，不能与“双侧错误均减少”的筛查目标混为一谈。

训练 OOF 参与了参数/阈值选择，其指标存在选择乐观偏差。新测试切分是本次搜索前固定的内部检查；原数据已经用于此前教学实验，也没有外部或时间外验证，所以不能称为全新的独立证据。LR/RF 是固定基线，只有 XGBoost 做了此次搜索，因此不是三个算法充分调优后的比较。

## 测试结果

下面所有行使用同一个 9,777 行测试集；主比较前三行都使用 0.50 阈值。

| 模型 | 阈值 | AUC | TN | FP（右上） | FN（左下） | TP | 总错误率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LR | 0.500 | 0.803703 | 5796 | 1919 | 621 | 1441 | 25.98% |
| RF | 0.500 | 0.863117 | 6812 | 903 | 665 | 1397 | 16.04% |
| XGBoost | 0.500 | 0.903643 | 7092 | 623 | 622 | 1440 | 12.73% |
| XGBoost：训练集选阈值 | 0.455 | 0.903643 | 6921 | 794 | 544 | 1518 | 13.69% |

在相同的 0.50 阈值下，XGBoost 相对 RF 的 FP 减少 280，FN 减少 43，AUC 提高约 0.0405。XGBoost 的 FN 比 LR 多 1，不能声称每一个错误数都优于每一个模型。

0.455 筛查阈值相对 XGBoost 的 0.50 阈值进一步减少 FN，但增加 FP 和总错误率；其两类错误仍低于 RF。该阈值作为另一种训练集选定的使用方式展示，并非通过测试集挑选。

旧版使用利率、评级，且测试样本切分不同。因此旧版更高的 AUC 不能代表同一审批前任务上的优势，也不能把两版 AUC 差异全部归因于去掉某一变量。

## 冻结的 XGBoost 参数

```python
xgb = XGBClassifier(
    n_estimators=1200,
    learning_rate=0.07415805824053914,
    max_depth=3,
    min_child_weight=4.369944120084866,
    subsample=0.8414902506684043,
    colsample_bytree=0.8702050470142777,
    reg_alpha=0.042260114069201264,
    reg_lambda=6.075700989710291,
    gamma=0.008911743386941978,
    scale_pos_weight=2.588304190635817,
    objective='binary:logistic',
    eval_metric='auc',
    tree_method='hist',
    random_state=42,
    n_jobs=2,
    verbosity=0,
)
```

## 图片与结果目录

图片位于 `../figures/`：

- `confusion_matrices_v6_050_600dpi.png` 和 `confusion_matrices_v6_050.pdf`：三模型统一 0.50 阈值，适合课程主比较。
- `confusion_matrices_v6_screening_600dpi.png` 和 `confusion_matrices_v6_screening.pdf`：XGBoost 使用训练集选定的 0.455 阈值，LR/RF 保持 0.50。
- `roc_curves_v6_600dpi.png` 和 `roc_curves_v6.pdf`：三模型 ROC/AUC。
- `threshold_cost_v6_600dpi.png` 和 `threshold_cost_v6.pdf`：训练 OOF 成本与错误数随阈值变化。

混淆矩阵 PNG 为 600 dpi、9600 × 3000 像素；只有一个共享色条，固定在最右侧。PDF 中的文字和线条为矢量；混淆矩阵色块按 600 dpi 保存。

本目录保存：

- `selected_params.json`：模型参数、阈值、数据 SHA256、切分规则、包版本。
- `xgb_trials.csv`：48 组参数及 5 折验证 AUC。
- `trial_*_oof.npy`：每个候选在训练集上的 OOF 分数，便于复核。
- `training_threshold_search.csv`：训练 OOF 筛查阈值搜索记录。
- `training_oof_predictions.csv`、`training_oof_metrics.csv`：训练 OOF 预测与指标。
- `test_predictions.csv`、`test_metrics.csv`：冻结模型的测试预测和指标。
- `cost_threshold_test_metrics.csv`：不同预先指定目标对应阈值的测试表现。
- `split_indices.npz`：训练/测试原始行索引。
- `encoded_feature_names.json`：实际模型的 15 列输入名称。
- `feature_availability_audit.csv`：变量时点和处理方式。
- `notebook_execution.log`：v6 全部代码单元的执行日志。

## 数据来源与复现

本地 `Programming_Session_2_Solution.ipynb` 明确将 `credit_dataset.csv` 描述为 synthetic。将原 CSV 的字段重命名、Y/N 映射为 1/0 后，本教程的 32,572 行全部可以在该源文件中逐行匹配（计入重复次数），源文件另有 9 行未进入教程 CSV。Bian 的 `process_data.ipynb` 使用另一份 `accepted_2007_to_2018Q4.csv.gz`，不能将其真实 LendingClub 数据说明套用于当前文件。

在 `imperial` 文件夹中运行：

```powershell
# 已完成本次搜索；需要复现搜索时再执行：
.\.venv_auc\Scripts\python.exe tune_preapproval_credit_risk.py --trials 48
# 不重新选参数，只用冻结参数复算：
.\.venv_auc\Scripts\python.exe tune_preapproval_credit_risk.py --refit
# 依据冻结参数重新生成、完整执行并验证 v6：
.\.venv_auc\Scripts\python.exe -X utf8 publish_preapproval_tutorial.py
```

课程 v6 本身不导入这两个脚本，也不读取搜索结果文件；与 `lending_club_tutorial.csv` 放在同一工作目录即可逐格运行。模型参数已嵌入 notebook。数据哈希不匹配时会要求重新训练，避免把旧参数选择记录套用到另一份数据。

本次运行环境：Python 3.12.14、numpy 2.5.3、pandas 3.0.5、scikit-learn 1.9.0、xgboost 3.4.1、optuna 5.0.0。不同库版本可能导致小幅结果差异。

验证：v6 所有代码单元执行通过；三模型的测试预测与调参脚本重新拟合结果逐项一致；notebook 重新计算的 XGBoost 训练 OOF 分数也一致。实际编码矩阵未包含利率、评级、当前贷款结果。
