# SCAPS-1D 对照参数清单（由 pvsim deck 生成）

- 层数：5（其中半导体 3 层，SCAPS 上限 7）
- 温度：300 K
- 入射功率：100 mW/cm²
- IV 扫描：0 → 1.8 V，37 点
- 电极：首个/末个 layer 视为接触层；边界类型 OHMIC，功函数 4.7 / 4.7 eV

## 逐层参数

| # | 名称 | 材料 | 厚度 (µm) | Eg (eV) | χ (eV) | εr | Nc (cm⁻³) | Nv (cm⁻³) | μn (cm²/Vs) | μp (cm²/Vs) | Nd (cm⁻³) | Na (cm⁻³) | τn (s) | τp (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | FTO | FTO | 0.2 | 3.7 | 4.2 | 9 | 1e+21 | 1e+21 | 20 | 10 | 1e+20 | 0 | 1e-09 | 1e-09 |
| 2 | TiO2 | TiO2 | 0.04 | 3.2 | 4 | 10 | 5e+20 | 5e+20 | 0.1 | 0.01 | 0 | 0 | 1e-09 | 1e-09 |
| 3 | MAPbI3 | MAPbI3 | 0.5 | 1.6 | 3.93 | 24 | 2.2e+18 | 1e+19 | 1.6 | 1.7 | 0 | 0 | 8e-09 | 8e-09 |
| 4 | Spiro | Spiro | 0.2 | 3 | 2.5 | 3 | 5e+19 | 5e+19 | 0.01 | 0.02 | 0 | 0 | 1e-09 | 1e-09 |
| 5 | Au | Au | 0.08 | 1 | 4.7 | 1 | 1e+22 | 1e+22 | 100 | 80 | 0 | 1e+20 | 1e-09 | 1e-09 |

## 填入 SCAPS 的位置

1. **Layer properties**：厚度、Eg、χ、εr、Nc、Nv、μn、μp 按上表；接触层（is_contact=1）填到最外侧 layer，并在 Contact 面板设功函数。
2. **Doping**：Nd → 施主浓度（N_D），Na → 受主浓度（N_A），其余填 0；类型按浓度非零项选择 n/p。
3. **Defects**：单能级 SRH 用 `set layerN.defect.singlelevel`，带隙中深处填 Et = Eg/2；由 τ 反推 Nt·σ：`τ = 1 / (σ · v_th · Nt)`，v_th ≈ 1e7 cm/s。本平台 taun/taup 为等效寿命，可直接令 σn=σp=1e-15 cm⁻² 后反解 Nt。
4. **Script 面板**：载入同名 `.script`（已按 pvsim 的 IV 设置写好）。

## 验证过的 SCAPS 层参数命令（可选，手填更快）

```text
set layer2.thickness 0.04     // µm
set layer2.Eg        3.2      // eV
set layer2.chi       4        // eV
set layer2.epsilon   10      
set layer3.thickness 0.5      // µm
set layer3.Eg        1.6      // eV
set layer3.chi       3.93     // eV
set layer3.epsilon   24      
set layer4.thickness 0.2      // µm
set layer4.Eg        3        // eV
set layer4.chi       2.5      // eV
set layer4.epsilon   3       
```

> 上面只列半导体层（3 层）；接触层（FTO、Au）在 SCAPS 里按电极/高掺杂层处理，功函数见顶部说明。
> 其余参数（Nc/Nv/迁移率/掺杂/缺陷密度）的命令名未在官方手册片段中确认，建议在 GUI 面板填写，避免脚本报错。
