# Khan 踝参数与紧凑库／完整 NPZ 训练

2026-11-09参考残差更新：以下两个训练启动器默认启用参考残差动作和隔离直通分支的goal噪声。
网络、优化器、数据加载与FK设置不变。新run及旧模式对照开关见
[参考残差与goal噪声](KHAN_REFERENCE_RESIDUAL.md)。
本次服务器实验的两个 `.sh` 启动器默认关闭执行器延迟；显式添加 `--actuator-delay` 可恢复0–20 ms延迟。

当前训练默认配置已在2026-11-09更新为0–20 ms actuator delay且reset首控制步立即执行、合并不足一个bin的尾部、物理reset至少剩余H32；reset抬高0.05m与首步action-rate保持不变。mixed480子集提取、全量训练及验证见 [Khan reset边界与采样](KHAN_RESET_SAMPLING.md)。以下踝参数及早期验证记录继续适用；新配置应新建run，恢复训练须匹配采样与机器人合同。

本轮按要求仅调整左右踝的标称执行器参数。其余关节、几何、质量与惯量、力矩限额、动作缩放、观测、奖励、DR 分布、ξ91、网络和优化参数保持不变。

| 双侧关节 | Kp | Kd | armature，kg·m² |
| --- | ---: | ---: | ---: |
| ankle pitch | 80 | 2 | 0.02 |
| ankle roll | 20 | 1 | 0.01 |

XML、profile、生成器及 provenance 同步更新。实际编译的 mjlab 模型已按关节名核对，其他23个执行器与修改前一致。原有每次 reset 的 0.8–1.2 倍随机化保留，故踝 pitch armature 为0.016–0.024，roll为0.008–0.012；pitch Kp/Kd为64–96/1.6–2.4，roll为16–24/0.8–1.2。

模型/profile 哈希随参数变更而改变。本轮应新建 run，不将旧物理配置的检查点作为相同模型继续 resume。动作库不需因此重新重定向：这些执行器参数不改变 FK 几何。

## 服务器脚本训练 `.pt`

先同步本轮代码和 `assets/robots/khan_mini/`。以下使用服务器上已确认的仓库和数据绝对路径：

```bash
cd /home/jaka/Workspace/SuperMimic-Reproduced-jaka
KHAN_LIBRARY=/home/jaka/Workspace/SuperMimic-Reproduced-jaka/data/pt/khan_combined_8452_compact.pt
```

推荐使用新启动器 `scripts/training/train_khan_compact_server.sh`。它与 NPZ 启动器调用同一个 `supermimic.scripts.train`，提供项目虚拟环境／机器人资产／库文件检查、环境变量清理、自动切换项目目录、已有检查点保护和 `--dry-run`。原 NPZ 启动器保持不变。

不开启参考 FK 编译（默认）：

```bash
bash scripts/training/train_khan_compact_server.sh \
  --library "$KHAN_LIBRARY" \
  --run-name khan_bones_pt_residual_goal_eager --num-envs 4096 \
  --no-reference-fk-compile
```

开启参考 FK 编译：

```bash
bash scripts/training/train_khan_compact_server.sh \
  --library "$KHAN_LIBRARY" \
  --run-name khan_bones_pt_residual_goal_compiled --num-envs 4096 \
  --reference-fk-compile
```

脚本默认库路径为项目下的 `data/pt/khan_combined_8452_compact.pt`，因此在该服务器目录布局下也可以省略 `--library`。`--library` 被转为训练入口的 `--data.library`，自动选择 compact；参考 FK 编译默认关闭。该开关同时控制环境当前参考和想象窗口参考的 FK，不改变预测机器人 FK、网络编译或优化器。服务器命令保留正式默认 H32、batch2048/4096、UTD0.4、320k学习率调度和关闭激活重计算的设置；编译不保证在目标 GPU 上提高完整训练吞吐。建议先用 eager 命令；对照时使用独立 run 名称。数据无需移动，也无需 NPZ 目录。

首次运行可先在上述命令末尾添加 `--dry-run`。它仅检查文件路径并打印实际命令，不执行 Python、不读取整库、不创建训练日志；库的张量／机器人 metadata 及恢复兼容性仍由实际训练加载器验证。库文件必须为可读非空 `.pt`。从其他目录调用脚本时，相对的 `--library`／`--resume` 路径按调用目录解析。

需要恢复同一数据／配置／资产的 run 时增加 `--resume logs/对应run/ckpt_latest.pt`。若 run 已有 `ckpt*.pt` 却未指定 `--resume`，脚本会拒绝启动。短时检查可加 `--stop-after-iterations 50`，它只限制执行步数，不改变320k学习率调度。

脚本清理 `PYTHONPATH`、`PYTHONHOME` 和遗留 torchrun 环境变量，设置 `PYTHONNOUSERSITE=1`、`MUJOCO_GL=egl`，并与 NPZ 脚本一样将 `OMP_NUM_THREADS`、`TORCHINDUCTOR_COMPILE_THREADS` 默认设为1。选卡继续使用 `CUDA_VISIBLE_DEVICES=0 bash ...`。`--help` 显示全部参数；`.pt` 使用 `--library`，NPZ 目录继续使用原脚本的 `--data-dir`。

## 直接播放紧凑 `.pt` 中的指定动作

`scripts/play_pt.py` 与 NPZ 的 `scripts/play.py` 使用相同的策略观测和查看器，直接使用训练/benchmark 的 compact reference FK，不需要先转 NPZ：

```bash
env -u PYTHONPATH -u PYTHONHOME MUJOCO_GL=glfw \
  .venv/bin/python scripts/play_pt.py \
  --ckpt logs/khan_bones_seed_pt/ckpt_0100000.pt \
  --library "$KHAN_LIBRARY" \
  --clips lafan1__fightAndSports1_subject4 \
  --num-envs 1 --device cuda:0 --viewer native
```

`--clips` 是 PT 中的精确动作名，保留 `lafan1__` / `mixed__` 前缀；可在同一参数后填写多个名字。加载时在 CPU mmap 源文件、按库顺序选取参考，仅将所选动作送到 GPU；保持源 tensor 的存储精度与数值，不写临时 PT、不做重采样、接地修正或 NPZ 转换。运行时按检查点的参考精度设置载入，由参考 FK 计算 body 状态。

默认 `--sampling start` 从首帧开始，单动作循环；选择多个动作时，每次 reset 都在所选集合内随机取动作，规则与 NPZ play 相同。播放使用无噪声、无 DR、无延迟的评估条件，保留检查点的 reset root z offset；默认显示蓝色半透明参考机器人。原 NPZ play 和训练默认加载路径不变。

无桌面的服务器改用 `MUJOCO_GL=egl` 和 `--viewer viser`，连接该机器的8080端口。参考 FK 默认 eager，可显式增加 `--reference-fk-compile`；单环境播放通常无需编译。`--steps 100` 可在执行100个控制步后自动关闭，适合播放 smoke test。

## 直接对紧凑 `.pt` 做 GPU benchmark

无需先导出 NPZ，可在服务器仓库根目录运行：

```bash
env -u PYTHONPATH -u PYTHONHOME OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl \
  .venv/bin/python -u scripts/eval_benchmark.py \
  --ckpt logs/khan_bones_pt_eager/ckpt_0100000.pt \
  --library "$KHAN_LIBRARY" --num-envs 2048 --device cuda:0 \
  --out logs/khan_bones_pt_eager/analysis/benchmark_0100000_pt.json
```

`--library` 与 `--npz-dir`／动作筛选参数互斥，评估库内全部动作。并行数按 `.pt` 内部动作数量确定，不把一个文件误当一条动作；默认1024，可指定2048等，显存预算由目标GPU决定。参考FK默认eager，可加 Python 入口的 `--reference-fk-compile` 开启。原 NPZ 的 `--npz-dir` 评估方式保持不变。

沿用原 benchmark 协议：每条动作首帧开始、使用确定性策略跑一次完整回合，关闭观测／动作／reset噪声、域随机化和执行器延迟，保留检查点的实际root reset offset。GPU接触动力学及后端数值仍可能有运行间差异，不承诺逐bit复现。`failed` 是训练的根高度／倾斜终止触发，不等同所有视觉失误。均值按动作等权，step-weighted按实际执行步数加权；摔倒后未执行的参考帧不补零，所以应同时看失败数和完成比例。它与训练时随机起点、难例采样和DR下的 `train_ret` 不是同一统计量。

输出结果JSON及同名 `.manifest.json`，记录数据／检查点SHA256、设备、实际并行数、参考FK设置、时间与协议。Torch显存计数不包含MuJoCo Warp分配，不能将该数值当作整张GPU的显存峰值。

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
