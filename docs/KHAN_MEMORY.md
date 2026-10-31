# 原始训练路径的12 GB显存测试

2026-10-31，NVIDIA GeForce RTX 4070 SUPER，设备可见显存11.59 GiB（标称12 GB）。
本轮只调整机器人并行环境数；激活重计算关闭，没有更改网络、学习batch、H、更新次数、
学习率、compile、域随机化、replay容量或数据集。沿用全部36段/396,990帧Khan动作。
这些历史测试使用anchor关闭的640维goal；当前Khan启动脚本默认增加anchor后为649维，
下面复现命令显式加 `--no-anchor-obs` 保持历史测试口径。

## 接触和约束容量

原G1环境在Git基线 `fc4095f` 的 `tracking_env_cfg.py:237` 明确使用
`nconmax=35, njmax=250`。Khan迁移时改为128/512是保守资源预留，**不是通过容量测试得出的值**，
也没有测得35/250对Khan一定不足。

进一步追溯发现35/250来自mjlab tracking模板；其中35在上游升级Warp时由旧150000预算改来，
250更早已经存在。参考迁移仓库的A3/X2使用256/1024，但公开资料同样没有最低容量测量依据。
完整commit来源与服务器启动方法见 [KHAN_SERVER_TRAINING.md](KHAN_SERVER_TRAINING.md)。

- `nconmax`：每环境平均接触预算。当前Warp分配所有环境共享的接触数组，
  总槽位为 `nconmax * num_envs`，单个环境可以超过平均预算。
- `njmax`：每环境独立约束行上限。接触法向/摩擦、关节限位等都会产生约束行；
  一个接触点可能对应多行，它不是机器人关节数。
- 例如1024环境，35/250对应共享35,840个接触槽位、每环境250行约束；
  128/512对应共享131,072个接触槽位、每环境512行约束。
- 更大容量会增加显存；容量足够时不会提高碰撞精度。容量不足时会丢弃超额接触/约束。
  确定安全容量需要测峰值及Warp overflow标志；仅“跑起来”不能证明原容量不足或当前容量最小。

本轮各环境数测试均保持Khan128/512，不将容量变化混入并行数对比。

## 固定训练参数

| 项目 | 全部测试相同 |
|---|---|
| WM hidden | 2048×3 |
| policy encoder/decoder hidden | 各2048/1024/512/256 |
| WM batch/window/batches | 2048 / 8 / 10 |
| policy batch/H | 4096 / 32 |
| critic | 5 epochs × 4 minibatches |
| UTD/warmup | 0.4 / 40步 |
| replay | 每环境1000行 |
| 激活重计算 | 关闭 |
| 计划执行 | 50控制步，保留正式320k学习率时间基准 |

## 实测结果

| 并行环境 | 结果 | 失败阶段 | 失败前Torch峰值allocated |
|---:|---|---|---:|
| 1024 | OOM | 第42步，首次actor前向 | 10.396 GiB |
| 32，前轮记录 | OOM | 第42步，首次actor前向 | 10.552 GiB |
| 1 | OOM | 第42步，首次actor前向 | 10.581 GiB |

峰值只表示失败之前已分配的Torch内存，不是完成训练的内存需求；也不包含Warp/驱动及其他程序。
各测试均完成初始化、40步预热和首次10批WM更新，随后在actor前向保存激活时失败。
因此初始化能放进显存，不代表能完成训练更新。

1024环境初始化后Torch allocated约1.155 GiB；1环境约0.533 GiB。
Replay中每环境静态分配约0.613 MiB，32降至1只减少约19 MiB replay；
而actor始终从replay有放回抽取4096个起点，构建32步可反传图，其主要激活量不随采集环境数下降。

**结论：在本机当前条件、当前实现、完整正式batch/H且关闭激活重计算的约束下，
仅调整并行环境数未找到可运行配置，1024不可以，1也失败。** 本轮没有再开启额外显存优化
或缩小batch来把失败改写为通过。此前32环境开启重计算的通过记录是另一种执行配置。

报告：

- `logs/khan_memory/results.json`：汇总。
- `logs/khan_original_1024/summary.json`、`logs/khan_memory/original_1024.log`。
- `logs/khan_original_1/summary.json`、`logs/khan_memory/original_1.log`。
- `logs/khan_formal_smoke/summary.json`：此前32环境未开启重计算的记录。

重复1024测试（当前12 GB机器预期OOM，换新run-name避免覆盖报告）：

```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
  uv run python scripts/smoke_train_khan.py --num-envs 1024 --steps 50 \
  --no-anchor-obs --no-activation-checkpoint --run-name khan_original_1024_retry
```
