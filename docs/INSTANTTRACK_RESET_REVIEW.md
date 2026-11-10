# InstantTrack v1.4 的 Evaluation-guided resets

2026-11-10，只读核查版本 `b9512a96d4e82e6ddea9a6de3b03276888bbcb2c`。本轮Khan仅新增Actor EMA与紧凑encoder，没有迁入本页的eprobe或改变现有G采样器。

## 是否替代G

**没有。** 裸`train.py`默认eprobe关闭；发布的`instanttrack` preset同时启用三条信号：

1. G：训练回合结束后更新其**起始bin**的回报难度。
2. fail_reset：训练条件下真实失败位置的统计，将一部分起点放在失败bin的前一bin。
3. eprobe：独立nominal仿真中的EMA策略失败位置，将一部分起点放在评估失败之前的窗口。

active sweep只决定下一轮评估哪些片段，不是第四种物理reset分布。

对B个bin，`h=softmax(-25*G/G_max)`，`m=.999*m+.001*h`，原底座为：

```
p_G = .5*Uniform(B) + .4*h + .1*normalize(m)
```

当训练失败与probe分布均有有效数据时，发布preset最终采用：

```
p_reset = .85 * (.7*p_G + .3*p_train_fail) + .15*q_probe
```

| 来源 | 最终名义份额 |
|---|---:|
| 均匀bin | 29.75% |
| 当前低G softmax | 23.8% |
| G难度分布历史 | 5.95% |
| 训练失败前一bin | 25.5% |
| 评估失败窗口 | 15% |

没有训练失败历史时，前面的fail_reset混合回退p_G；首轮尚无probe失败时focus_share为0。极小训练失败历史的归一化floor及multinomial最终重归一化可使名义份额略变。

eprobe自身找失败只用物理模拟器和EMA actor，不需要G、critic或WM；但发布实现继续依赖G决定其余reset起点，并持续更新G。uniform+eprobe可另做实验，不能当成原库已经验证的替代方案。

## 一轮probe具体怎么运行

- 每条clip按300帧（6秒）划分tile anchor；原40条库共1489 tiles。
- v1.4最多选372个tile：最多一半给失败score最高者，其余选最久未测试者。初始age随机打散。
- 整轮共抽一次整数lead，均匀取200–450帧（4–9秒）。每个起点为 `max(tile-lead, clip_start)`，终点为 `min(tile+300, clip_last)`，因此通常滚动10–15秒。
- 在独立、无DR、无reset/action/obs噪声的环境中，从该帧的参考root pose、root速度、关节位置/速度初始化；previous action归零，history重复初始观测。没有复制训练环境的实际状态。
- 用部署EMA策略闭环运行。异步模式开始时复制一次EMA参数和buffers，整轮使用固定快照，每个训练iteration推进10个probe步。
- 记录首次物理高度/倾斜termination：高度误差>.25m或重力方向指标差>.8（不是直接比较.8弧度）。物理termination比较post-state对f+1，存入的failure frame是pre-transition f。
- 无需等待完整大动作从首帧播放到失败处。lead-in提供一段闭环准备过程，但这是中途参考reset测试，不等价于整条clip首帧benchmark。

非有限action也计入失败，但其异常分支不会立刻设置done，连续nonfinite时可能覆盖fail_f；“首次失败”对正常物理termination成立，对该异常分支不完全成立。

## 如何由失败位置生成reset概率

每扫结束，逐bin计数更新 `F=.7*F+本轮首次失败计数`，再将质量向前传播：

```
q[b] ∝ F[b] + F[b+1] + ... + F[b+10]  # 仅同一clip
```

所以代码实际包含**失败所在bin本身和前10个bin，最多11个1秒bin**。窗口不会跨clip，但失败bin后半段也可能成为起点，不能说每个起点都严格早于实际失败时刻。靠近clip开头的失败获得较短窗口，同一次失败贡献的总未归一化质量也更少。

从混合后的bin分布采bin，再在该bin的可用frame范围采起点；最终训练物理reset仍按原训练条件设置参考状态及噪声/DR，并非把probe世界状态复制过来。

## 默认参数与持续聚焦

发布preset：从iteration1000开始，每125次iteration发起一轮；tile300、lead200/450、窗口10、failure decay.7、active372、async10、focus share.15、fail_reset.3、policy EMA.995。

裸train.py默认share0、active0（全扫）、every250、start1250、decay.5，不能把“当前版本包含功能”理解成裸命令默认启用。

**零失败不自动撤回15%聚焦。** 一轮没有新失败时，F仅整体乘.7，归一化后的q基本不变；即使F下溢成0，代码跳过set_focus，旧q/share仍可能保留。score也按衰减失败计数而非失败率更新，active频繁重测的片段暴露次数更多。oldest配额保护覆盖，但不代表无偏估计。

## 对Khan的判断

优势是额外取得真实模拟器中部署策略的失败位置，不受G的续尾数值直接支配；向前窗口有助于学习准备过程，active降低评估开销。

但eprobe不修改G的目标、按批EMA、softmax温度和历史更新，因此不会修复已有G估计的批次频率偏置。我们已经有重复bin平均，参考的bin_fix不等于同bin计数加权。

若在当前p_G上仅加入15%eprobe，一个原40%的单bin仍可能占34%；完整参考还引入25.5%训练失败来源，三条困难信号可能指向同一区域，不能保证总概率更分散。

8452库按相同规则有16750 tiles，372仅覆盖2.22%/扫；无hot约46扫覆盖、half-hot约91扫。不能照抄原库“几扫覆盖”的预算。10秒前向窗口还可能覆盖我们多数短clip的大部分长度。

我们的整尾bin合并、H32合法物理reset范围需要保留。probe失败frame要映射到我们的bin表并投影到合法起点；不能原样复制参考bin编号和尾部边界。

现有整片nominal benchmark0失败与随机中途reset是不同条件；反过来，不触发termination的脚抖、单腿高度不足，也无法单靠failure-only probe发现。应同时记录跟踪与足部平滑性。

没有发现同配置仅开关eprobe_share的发布消融。active预算存在开发对照：5seed的全扫/active372平均持续达标2.772/2.669分钟，但只有2/5seed改善，且并非最终完整encoder梯度配置。v1.4四项联合变化的15配对seed结果p=.10，不能把整体收益单独归于eprobe。

适合后续独立实验，需明确覆盖周期、focus撤回规则、H32边界和混合份额；本次没有将其加入生产代码。

## 源码依据

- [preset完整参数](https://github.com/Huy-Quang-Dao/instantTrack/blob/b9512a96d4e82e6ddea9a6de3b03276888bbcb2c/scripts/presets.sh#L20)
- [G、fail_reset与focus混合](https://github.com/Huy-Quang-Dao/instantTrack/blob/b9512a96d4e82e6ddea9a6de3b03276888bbcb2c/src/instanttrack/env.py)
- [probe调度、tile选择、失败窗口](https://github.com/Huy-Quang-Dao/instantTrack/blob/b9512a96d4e82e6ddea9a6de3b03276888bbcb2c/src/instanttrack/train.py)
