# Khan 踝参数与紧凑库／完整 NPZ 训练

本轮按要求仅调整左右踝的标称执行器参数。其余关节、几何、质量与惯量、力矩限额、动作缩放、观测、奖励、DR 分布、ξ91、网络和优化参数保持不变。

| 双侧关节 | Kp | Kd | armature，kg·m² |
| --- | ---: | ---: | ---: |
| ankle pitch | 80 | 2 | 0.02 |
| ankle roll | 20 | 1 | 0.01 |

XML、profile、生成器及 provenance 同步更新。实际编译的 mjlab 模型已按关节名核对，其他23个执行器与修改前一致。原有每次 reset 的 0.8–1.2 倍随机化保留，故踝 pitch armature 为0.016–0.024，roll为0.008–0.012；pitch Kp/Kd为64–96/1.6–2.4，roll为16–24/0.8–1.2。

模型/profile 哈希随参数变更而改变。本轮应新建 run，不将旧物理配置的检查点作为相同模型继续 resume。动作库不需因此重新重定向：这些执行器参数不改变 FK 几何。

## 服务器直接训练 `.pt`

先同步本轮代码和 `assets/robots/khan_mini/`。以下使用服务器上已确认的仓库和数据绝对路径：

```bash
cd /home/jaka/Workspace/SuperMimic-Reproduced-jaka
KHAN_LIBRARY=/home/jaka/Workspace/SuperMimic-Reproduced-jaka/data/pt/khan_combined_8452_compact.pt
```

不开启参考 FK 编译：

```bash
env -u PYTHONPATH -u PYTHONHOME OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl \
  .venv/bin/python -u -m supermimic.scripts.train \
  --robot khan_mini --run-name khan_bones_pt_eager \
  --data.library "$KHAN_LIBRARY" --env.num-envs 4096 \
  --env.no-reference-fk-compile
```

开启参考 FK 编译：

```bash
env -u PYTHONPATH -u PYTHONHOME OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl \
  .venv/bin/python -u -m supermimic.scripts.train \
  --robot khan_mini --run-name khan_bones_pt_compiled \
  --data.library "$KHAN_LIBRARY" --env.num-envs 4096 \
  --env.reference-fk-compile
```

`--data.library` 自动选择 compact，默认不开启参考 FK 编译。该开关同时控制环境当前参考和想象窗口参考的 FK，不改变预测机器人 FK、网络编译或优化器。服务器命令保留正式默认 H32、batch2048/4096、UTD0.4、320k学习率调度和关闭激活重计算的设置；编译不保证在目标 GPU 上提高完整训练吞吐。建议先用 eager 命令；对照时使用独立 run 名称。命令清除外部 `PYTHONPATH`/`PYTHONHOME`，使用本项目 `.venv`；数据无需移动，也无需传 `--data.npz-dir`。

## 从 `.pt` 导出完整训练 NPZ

```bash
env -u PYTHONPATH -u PYTHONHOME OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/data/library_to_npz.py \
  --library "$KHAN_LIBRARY" \
  --out-dir data/npz/khan_mini/bones_combined_8452 \
  --robot khan_mini --device cpu --chunk-size 8192
```

默认转换全部动作；输出目录须为新目录或空目录，禁止覆盖。可用 `--device cuda:0` 执行分块 GPU FK。小批量可加 `--limit 3`，或按实际名称使用 `--clips 名称1 名称2 名称3`。默认保存当前训练所需的15个跟踪body；如需和旧 PKL 转换数据一样保存全部45个body，可加 `--all-bodies`。

输出是每条动作一个50 Hz、float32 NPZ，含 `joint_pos`、`joint_vel`、四项 `body_*_w`、`joint_names`、`body_names`、`fps` 和来源信息。它是完整训练格式，区别于只保存 `q` 的预览 NPZ。训练时 body 数组一次载入 GPU，以后直接索引，不再在线执行参考 FK。

导出直接保留 `.pt` 已存储的数值，不重采样、不重新差分速度、不抬高 root、不裁剪关节。fp16 之前已损失的精度不能恢复。为避免 FK 的根四元数归一化和角速度坐标变换改变参考输入，新增有格式标记的 `reference_state`，保留原67维参考；body数组则保存原compact FK的规范化输出。加载器严格验证它与标准数组的一致性，仅对带该扩展的新文件读取原参考状态，旧NPZ行为保持不变。这里的实际库是fp16；若未来库以fp32存储，导出保留fp32，比较compact时也应显式使用fp32存储设置。

转换同时生成 `manifest.csv` 和 `conversion_report.json` 便于核对。普通训练会发现目录内全部 `.npz`，无需修改或显式传入这两个文件。

## 用导出的完整 NPZ 训练

```bash
env -u PYTHONPATH -u PYTHONHOME OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl \
  .venv/bin/python -u -m supermimic.scripts.train \
  --robot khan_mini --run-name khan_bones_full_npz \
  --data.npz-dir data/npz/khan_mini/bones_combined_8452 \
  --env.num-envs 4096 --env.no-compact-library
```

不要同时传入 `--data.library`，因为它优先于 NPZ 目录。完整模式下参考 FK 编译开关不生效；想象中世界模型预测状态的可微 FK 仍正常运行。

本库3818945帧采用完整模式时，参考／15body／基础goal核心GPU数组约4.98 GB，而compact参考张量约0.512 GB；均未计索引、replay、网络及临时张量。

## 本轮小批量与实际训练验证

从实际8452条紧凑库选择以下三个来源，合计6942帧：

| 动作 | 来源 | 帧数 |
| --- | --- | ---: |
| dance_dip_001__A467 | BONES-SEED | 101 |
| lafan1__dance1_subject1 | LAFAN1 | 6574 |
| mixed__00_self_pico_record_087 | 旧 mixed | 267 |

实际转换与训练命令：

```bash
OMP_NUM_THREADS=1 .venv/bin/python scripts/data/library_to_npz.py \
  --library "$KHAN_LIBRARY" --out-dir data/npz/khan_mini/bones_smoke \
  --robot khan_mini --device cpu --chunk-size 257 \
  --clips dance_dip_001__A467 lafan1__dance1_subject1 mixed__00_self_pico_record_087

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl \
  TORCHINDUCTOR_COMPILE_THREADS=1 .venv/bin/python scripts/smoke_train_khan.py \
  --data-dir data/npz/khan_mini/bones_smoke \
  --num-envs 32 --steps 50 --activation-checkpoint \
  --run-name khan_full_npz_smoke
```

本机实际输出位于 `logs/pt_to_npz_ankle_alignment_20261107/npz/`，训练run为 `logs/khan_pt_to_npz_new_ankles_smoke/`。本机12 GB测试开启激活重计算，其余网络、batch、H32、优化器及学习率调度保持正式值。训练完成40次WM、4次策略、80次critic更新，参数及已记录标量有限，模型／优化器／采样器检查点恢复通过；加载报告确认 `source_type=npz_directory`、`compact=False`、3条／6942帧。

CPU全帧对照中，原始67维参考、155维actor goal、critic参考特征和四项body数组最大差均为0。19帧独立MuJoCo对照的位置、线速度、角速度最大差分别约 `5.98e-7 m`、`3.13e-6 m/s`、`6.30e-6 rad/s`。输入PT的SHA256不变。转换30项测试、资产10项测试、相关运行时41项回归均通过。

补充 GPU 全6942帧对照，比较 CPU 导出的预计算 NPZ 与实际 compact CUDA FK：

| 最大绝对差 | CUDA eager | CUDA compiled |
| --- | ---: | ---: |
| body位置，m | 9.537e-7 | 9.537e-7 |
| 四元数分量 | 6.557e-7 | 7.749e-7 |
| 线速度，m/s | 5.722e-6 | 6.438e-6 |
| 角速度，rad/s | 1.622e-5 | 1.622e-5 |

这是 FP32 运算顺序造成的数值容差内一致，不宣称 CPU/GPU 逐 bit 相同。GPU上的原参考、actor goal和critic参考特征仍逐值相同。另实际创建4个mjlab环境，核对训练命令的GPU参考body缓存，最大位置差 `2.385e-7 m`；这里比较参考缓存，而不是要求物理机器人跟踪误差为零。完整NPZ训练读取GPU预计算数组，compact训练才在线做参考FK；两者的预测机器人可微FK仍正常在GPU计算。

还实际执行了 `--device cuda:0` 的101帧单动作转换，CLI和加载验证通过；与CPU导出的关节及原参考逐值一致，body数据处于上述FP32容差范围。测试原始PT和CPU导出NPZ的哈希均保持不变。

完整验证报告见 `logs/pt_to_npz_ankle_alignment_20261107/independent_validation.json`、`independent_validation_gpu.json`、`cuda_conversion_validation.json` 和 `training_smoke_summary.json`。这验证转换、加载和训练更新流程，不代表50步策略已收敛。
