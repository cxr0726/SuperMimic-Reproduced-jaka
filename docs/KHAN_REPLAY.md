# Khan NPZ 运动学重播与足底穿地检查

本工具将已转换 NPZ 的根位置/四元数、关节位置/速度逐帧写入真实 mjlab 场景，
调用 `sim.forward()` 和 `scene.update()` 刷新几何，不调用底层 `env.step()` 或
`sim.step()`。它不执行策略、不做 PD 跟踪、不积分、不抬高根轨迹，不修改源 NPZ。
因此看到的是参考动作本身与当前模型、世界地面 `z=0` 的相对位置。

## 原生窗口重播

在仓库根目录执行：

```bash
env -u PYTHONPATH uv run python scripts/replay_khan_npz.py \
  --npz data/npz/khan_mini/lafan1/walk1_subject1.npz \
  --viewer native --geometry both --loop
```

默认暂停。空格播放/暂停，左右方向键逐帧并暂停，Home 回到选定窗口首帧。
`--no-paused` 可直接播放，`--speed 0.5` 为半速，`--geometry visual|collision|both`
切换显示内容。窗口左上角显示原始帧号、时间、左右脚碰撞底与视觉底高度（mm）。
低于 `--tolerance-m` 阈值的脚标红，地面半透明以便看到地下几何。

检查全数据最深穿地附近的 2 秒片段：

```bash
env -u PYTHONPATH uv run python scripts/replay_khan_npz.py \
  --npz data/npz/khan_mini/lafan1/dance2_subject2.npz \
  --viewer native --start-frame 8600 --end-frame 8700 --loop --geometry both
```

`--end-frame` 是不包含的右边界。初始位置从 NPZ 原样读取，172.98 秒即源帧 8649。

浏览器查看器带 Source frame 滑条，打开程序打印的 `http://127.0.0.1:8080`：

```bash
env -u PYTHONPATH uv run python scripts/replay_khan_npz.py \
  --npz data/npz/khan_mini/lafan1/dance2_subject2.npz \
  --viewer viser --start-frame 8600 --end-frame 8700 --loop
```

## 无窗口导出

```bash
env -u PYTHONPATH uv run python scripts/replay_khan_npz.py \
  --npz data/npz/khan_mini/lafan1/dance2_subject2.npz \
  --headless --start-frame 8600 --end-frame 8700 --max-frames 100 \
  --geometry both --video replay.mp4 --output-dir logs/khan_replay/dance2_worst
```

输出：

- `first_frame.png`：选定窗口首帧。
- `worst_collision_frame.png` / `worst_visual_frame.png`：窗口内最深穿地帧。
- `replay.mp4`：指定帧范围的视频；速度通过帧率控制，不改变数据。
- `ground_audit.json`：该动作全帧几何审计。
- `replay_summary.json`：重播范围、最深帧、forward 次数及 `physics_integration_steps=0`。

不需要视频时省略 `--video`，仍会保存关键帧。`--max-frames` 约束 headless 导出帧数或
交互窗口推进步数；交互窗口默认暂停时需播放才推进。关闭窗口或 Ctrl+C 结束。

## 精确足底审计

```bash
# 全部36段，每一帧，1 mm阈值；只用CPU运动学
env -u PYTHONPATH uv run python scripts/data/audit_khan_ground.py \
  --npz-dir data/npz/khan_mini/lafan1 --tolerance-m 0.001 \
  --output logs/khan_replay/ground_audit.json \
  --per-frame-csv logs/khan_replay/ground_heights.csv

# 单段，或用 --stride 10 快速抽查（默认stride=1才是全帧）
env -u PYTHONPATH uv run python scripts/data/audit_khan_ground.py \
  --npz data/npz/khan_mini/lafan1/walk1_subject1.npz
```

判定使用实际几何底部，不使用 ankle link 原点：

- 碰撞底：每只脚 7 个 capsule，最低点精确为
  `center_z - abs(axis_world_z) * half_length - radius`，再取所有足底几何的最小值。
- 视觉底：ankle_roll 及其固定焊接组件的全部脚部 visual，每只脚 4 个 mesh，
  用 MuJoCo 编译后顶点和实际 geom 世界变换求最低点。用于极值计算的凸包简化不改变最低高度。
- “穿地帧”指至少一个脚底 `z <= -0.001 m`；负高度总数也独立记录。
  碰撞底与可见网格底分开统计，避免二者厚度差异造成混淆。

当前模型的 home 姿态足碰撞底约为 **+1.94966 mm**，用于确认坐标与阈值没有误报。

## 实测结果

在当前 Khan 模型、地面 `z=0`、没有根高度补偿的情况下：

| 项目 | 碰撞几何 | 可见脚部网格 |
|---|---:|---:|
| 全量检查 | 36 段 / 396,990 帧 | 相同 |
| 至少一脚穿地超过 1 mm 的帧数 | 385,191（97.0279%） | 385,926（97.2130%） |
| 存在穿地的动作数 | 36 / 36 | 36 / 36 |
| 最低高度 | −92.89645 mm | −94.84461 mm |
| 最深动作 / 帧 / 时间 | dance2_subject2 / 8649 / 172.98 s | 相同 |

`walk1_subject1` 全部 13,065 帧至少一脚碰撞底低于 −1 mm，最深约 **−53.41897 mm**，
发生在第 3239 帧（64.78 s）。数据格式与 FK 数值校验通过不等于接地标定通过；
当前参考动作与模型/地面零点存在明显接地偏差。本工具仅暴露问题，不自动修复动作。

验证包括 13 项几何底高测试，以及真实 GPU 重播集成测试：将底层环境和仿真器的 step
替换成抛错函数后，seek/逐帧/循环仍通过，GPU 仿真时钟不变，NPZ 哈希不变。
100 帧 headless/MP4、原生窗口和 Viser 有限帧重播均已实测通过；记录位于 `logs/khan_replay/`。
没有运行训练。

## 与G1训练模型的关系

非足底摩擦现为 **1.0**，足底保持 **0.6**，与当前 mjlab G1 相同；EntityCfg、
BuiltinPositionActuator、0.9 软限位、视觉/碰撞分组、足底接触规则及动作处理类型基本对齐。
机器人自身的关节数、几何、增益、初态和动作缩放仍按 Khan 配置。

完整训练端仍有 G1 的机器人选择、pelvis/torso 名称、29 关节、FK 及状态/观测维度假设，
尚未接通 Khan。接地标定、Khan 训练端迁移仍需后续处理。

## Git 内容

代码、文档与独立转换的 `assets/robots/khan_mini/` 资产可纳入版本控制。
`ref_thirdparty/` 已在 `.gitignore` 中排除；原始/转换数据、日志、截图和视频也按现有
`data/`、`logs/` 规则保留在本地，不纳入提交。
