# Cs2AgBiBr6 参考电池（复刻 Silvaco 输入卡 test0408，吸收层 800 nm）
#
# 参数来源：D:\王继凯毕设资料\原始数据\test0408_3D_TiO2_NiO_loop_perov.in
# 结构：FTO 20 / TiO2 20 / Cs2AgBiBr6 800 / NiO 20 / Au 100 nm
# 参考数据：D:\王继凯毕设资料\原始数据\钙钛矿层厚度扫描\
#           solarex_TiO2_NiO_perov_0.8um.log（Atlas SSF，面积 1e-6 cm²）
#           汇总：..\summary_results_TiO2_NiO_perov_scan.csv
#           0.8 um 参考（标准 1 sun）：
#             Jsc 27.26 / Voc 1.1821 / FF 0.809 / PCE 26.06%
# 注意：datasets\jv_800nm.xlsx（Jsc 15.04）是另一轮数据，不要用它做基准。
#
# 本地 Atlas 复现（decks\atlas2d_800nm.in + scripts\atlas_reference.py）：
#   同一结构、同材料/陷阱参数，2D 1 sun 得
#     Jsc 27.27 / Voc 1.1822 / FF 0.810 / PCE 26.13%
#   与上述 3D 结果四位有效数字吻合 → 参考数据确为 1 sun，可用作基准。
#
# 消融实验（Atlas 2D，逐项删除）：
#   * 去 TRAP（1e14 @0.8 eV）：Voc 仅 +0.5 mV、FF +0.002 → 与寿命模型等价，
#     本平台不必实现显式陷阱；
#   * 去 copt=2.3e-9：Jsc 30.56 / Voc 1.2340 / FF 0.879 → 该项影响 Voc 52 mV、
#     FF 0.069。copt 是 Silvaco 的辐射复合系数（optr 模型，R=B(np-ni^2)，
#     2.3e-9 cm^3/s 是直接带隙材料的典型值）；原始 3D 卡第 128-129 行确实
#     启用了 optr。本平台已实现该物理项：内核 pvk_layer.b_rad_cm3_s /
#     deck 关键字 copt（0=关闭，旧卡逐位不变）。
#
# 电学侧残差：
#   * 补上辐射复合后 Voc 1.1816（Atlas 1.1822）、FF 0.8067（0.810）。
#     此前无 copt 时的 +45 mV 基线偏高，主要就是缺 B·n·p：Voc 附近 np 大，
#     辐射项把载流子密度压回来。
#   * 低光收敛（已解决）：根因是 B(x)=x/(e^x-1) 用 exp(x)-1 求值，|x| 略大于
#     1e-8 时有 1e-8 的相对误差，而 SG 通量是 n·B(±Δ) 大项相消，这个误差
#     直接成了 div J 的精度地板。改用 expm1 后 1 sun–0.01 mW 五档均
#     0–1.2 V 全部收敛、Jsc 严格线性（见 benchmark 第 10 项）。
#
# 光学侧修正（本轮）：
#   * 旧实现把背反射率 r_back=0.5 用在每一程结束：内部光从前表面反射时
#     也被当成金属镜面，且金属界面用实折射率算透射，重复/错误扣能量，
#     q∫G 只有 28.36 mA/cm²。现在正向到背电极乘金属反射率（末层有 nk
#     时按复折射率、从相邻半导体侧入射；缺 nk 时按 Au 典型值 0.95），
#     反向到前表面乘菲涅尔反射率，q∫G=30.59，与 Atlas 去 copt 的
#     基线 Jsc=30.56 一致。
#   * 剩余 Jsc -4.1% 不是生成差，而是复合损失偏大（本卡 4.44 vs
#     Atlas 3.29 mA/cm²）：短路段吸收层的准费米劈裂略高于 Atlas。
#
# 已知差异（对照时需记住）：
#   * Atlas 用 index.file 的 n,k 做光学（含亚带隙吸收，无带隙截断）；
#     本平台同样走 nk，且此处显式关掉 bandgap.mask 以对齐 Atlas 口径；
#   * Atlas 另有 auger(1e-32) 与两个界面陷阱，实测贡献 ≈0；
#   * Atlas 的 gold 是 conductor（无 Au region）；本平台按 100 nm Au
#     接触层处理，光学上由 r_back 的金属反射率代表；
#   * 强注入区（>1.36 V）Newton 仍会停滞；JV 指标只用 Voc 以下的数据。
# 当前结果：Jsc 26.15 / Voc 1.1816 / FF 0.8067（对照 Atlas：Jsc -4.1%、
# Voc -0.6 mV、FF -0.0033）。

title Cs2AgBiBr6 800nm reference cell (Silvaco replica)

mesh nodes=201 interface.refine=1.0 space.mult=1.0 temperature=300

region num=1 material=FTO       thickness=20
region num=2 material=TiO2      thickness=20
region num=3 material=Cs2AgBiBr6 thickness=800
region num=4 material=NiO       thickness=20
region num=5 material=Au        thickness=100

doping region=1 n.type conc=1.0e18
doping region=2 n.type conc=5.06e19
doping region=3 p.type conc=4.0e16
doping region=4 p.type conc=1.0e18
# 本平台约定：电极层的掺杂决定接触极性（相当于 Atlas 的功函数选择）。
# Au 不掺时接触电位会钉在本征能级，NiO 侧出现 ~1.8 V 假势垒并导致发散。
doping region=5 p.type conc=1.0e20

# 材料参数逐项照抄 Silvaco 卡片
material name=FTO eg=3.5 chi=4 eps=9 nc=2.2e18 nv=1.8e19 \
         mun=20 mup=10 taun=1e-5 taup=1e-6

material name=TIO2 eg=3.2 chi=4 eps=100 nc=2e20 nv=1e21 \
         mun=6e-3 mup=6e-3 taun=1e-6 taup=1e-5

material name=CS2AGBIBR6 eg=1.61 chi=3.72 eps=5.8 nc=2e18 nv=1e19 \
         mun=9.28 mup=9.28 taun=6.6e-7 taup=6.6e-7 copt=2.3e-9

material name=NIO eg=3.8 chi=1.46 eps=10.7 nc=2.8e19 nv=1e19 \
         mun=12 mup=2.8 taun=1e-6 taup=1e-5

models srh optr
optics spectrum=am15g intensity=100 bounces=1 bandgap.mask=0 \
       nk.dir="D:/交互界面_data/datasets/nk"

method newton maxiter=80 tol=1e-8 damping=1.0
solve vstart=0 vfinal=2.0 vstep=0.02

output fields=X,PSI,N,P,EFIELD,JTOTAL,CB,VB run=cs2agbibr6_800nm
