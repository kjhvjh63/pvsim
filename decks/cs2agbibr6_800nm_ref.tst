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
#   * 去 copt=2.3e-9：Jsc 30.93 / Voc 1.2340 / FF 0.879 → 该项影响 Voc 52 mV、
#     FF 0.069。copt 是 Silvaco 的辐射复合系数（optr 模型，R=B(np-ni^2)，
#     2.3e-9 cm^3/s 是直接带隙材料的典型值）；原始 3D 卡第 128-129 行确实
#     启用了 optr。本平台已实现该物理项：内核 pvk_layer.b_rad_cm3_s /
#     deck 关键字 copt（0=关闭，旧卡逐位不变）。
#
# 电学侧残差：
#   * 补上辐射复合后 Voc 1.1820（Atlas 1.1822）、FF 0.8066（0.810）。
#     此前无 copt 时的 +45 mV 基线偏高，主要就是缺 B·n·p：Voc 附近 np 大，
#     辐射项把载流子密度压回来。
#   * 低光收敛（已解决）：根因是 B(x)=x/(e^x-1) 用 exp(x)-1 求值，|x| 略大于
#     1e-8 时有 1e-8 的相对误差，而 SG 通量是 n·B(±Δ) 大项相消，这个误差
#     直接成了 div J 的精度地板。改用 expm1 后 1 sun–0.01 mW 五档均
#     0–1.2 V 全部收敛、Jsc 严格线性（见 benchmark 第 10 项）。
#
# 光学侧修正：
#   * 背反射路径（非相干，默认）：旧实现把 r_back=0.5 用在每一程结束（反向程
#     回前表面也当金属镜面），金属界面又用实折射率重复扣透射，q∫G 只有 28.36。
#     改为正向乘金属反射率（末层有 nk 时按复折射率，缺 nk 时按 Au 典型
#     0.95）、反向乘前表面菲涅尔，q∫G 提到 30.59。
#   * nk 外推：FTO.txt 止于 0.898 µm。单色 Atlas 实验（0.86–0.92 µm）
#     显示该处无台阶（A(0.90)/A(0.89)=0.953，端点钳制预测 0.952，而
#     √εr 回退预测 0.766），说明 Atlas 做平滑外推。内核已改为把 n 钳制
#     到表端点，消除 898 nm 处 25% 的假前反射，q∫G 30.59→31.44。
#   * 相干薄膜光学（optics 的 coherent 开关）：逐层复折射率 TMM 求内部场，
#     尾部 Au 接触层不参与传播、其界面按完美背镜 E=0；吸收密度 n·k·|E|²
#     按网格节点沉积。默认关闭（旧卡逐位不变），本卡启用：
#       q∫G 31.56；Jsc 27.11（Atlas 27.27，-0.6%）——这是「完美背镜」理想化。
#   * 背电极改用真实 Au：cfab/Au_Formatted.txt 经 scripts/convert_nk.py 规范成
#     nk/Au.txt 后，相干栈包含 100 nm Au（背面空气），不再用完美背镜：
#       A(0.8/0.9/1.0 µm)=0.997/0.511/0.297；q∫G 31.39（更接近 Atlas 真实
#       ~31.36）；Jsc 26.98（-1.09%）、Voc 1.1822、FF 0.8092、PCE 25.81%。
#     若 nk.dir 下没有 Au.txt，自动回退完美背镜（上一条的 31.56 / 27.11）。
#
# 短路复合直读（V=0，1 sun，PVK 内积分；Atlas 加 output u.radiative u.srh u.auger）：
#   Atlas：Rad 4.022 / SRH 0.063 / Auger≈0；本卡相干（含 Au）：Rad 4.301 / SRH 0.070。
#   复合参数（B、τ、ni）口径一致；差 +0.28（+7.0%）等价于准费米劈裂高 1.6 mV，
#   来自短路段载流子密度（接触/输运），不是光学。剩余 −1.09% 的分解：
#   生成 +0.17 + 辐射 +0.28 → 净 −0.11 ~ −0.16。
#
# 单色口径与 Atlas 光学量说明：
#   * Atlas 的 BEAM 是非相干光线追踪：解文件里 O 行之后的 R 光线表给出光滑的
#     A(λ)（A(800)=0.663、A(900)=0.409、A(1000)=0.242），且 400–600 nm 的
#     A_all 正好是 1.000——光线表不扣前表面反射。q∫A·φ = 31.22 mA/cm²，
#     与本文两种模型的 31.44（非相干）/ 31.56（相干）只差 0.5–1.2%。
#   * 用 beam 单色、V=0 电流/qΦ 反推的 A(800)≈0.49 不能当吸收率：它把短路段
#     复合与陷阱损失一起算了进去（0.1 W/cm² 单色下损失可达 10 mA/cm² 量级）。
#   * 因此逐波长对照以 R 光线表为准；相干模型的驻波条纹只在宽带积分里抵消，
#     不应要求与 Atlas 逐点重合。复跑：scripts/atlas_optics.py --compare。
#   * 复现对照：optics 的 atlas.ray 开关按「单程、无界面反射」逐点复刻光线表，
#     A(0.8/0.9/1.0 µm)=0.654/0.394/0.224 vs Atlas 0.663/0.409/0.242，RMS 0.012；
#     剩余小差来自 Atlas 阴极侧约 5% 的弱背反射。该模式只用于核对，不做物理预测。
#   * Atlas 去 copt 的 Jsc 复测为 30.93（旧记录 30.56 有误）。
#
# 已知差异（对照时需记住）：
#   * Atlas 用 index.file 的 n,k 做光学（含亚带隙吸收，无带隙截断）；
#     本平台同样走 nk，且此处显式关掉 bandgap.mask 以对齐 Atlas 口径；
#   * Atlas 另有 auger(1e-32) 与两个界面陷阱，实测贡献 ≈0；
#   * Atlas 的 gold 是 conductor（无 Au region）；本平台用真实 100 nm Au 的
#     复折射率（nk/Au.txt），非相干路径的 r_back 也由它逐波长算出；
#     缺 Au.txt 时相干回退完美背镜、非相干回退 0.95。
#   * 强注入区（>1.36 V）Newton 仍会停滞；JV 指标只用 Voc 以下的数据。
# 当前结果（coherent + Au）：Jsc 26.98 / Voc 1.1822 / FF 0.8092 / PCE 25.81%
#           （完美背镜：Jsc 27.11 / Voc 1.1823 / FF 0.8092 / PCE 25.94%；
#            非相干：Jsc 26.82 / Voc 1.1821 / FF 0.8065 / PCE 25.57%）

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
optics spectrum=am15g intensity=100 bounces=1 bandgap.mask=0 coherent \
       nk.dir="D:/交互界面_data/datasets/nk"

method newton maxiter=80 tol=1e-8 damping=1.0
solve vstart=0 vfinal=2.0 vstep=0.02

output fields=X,PSI,N,P,EFIELD,JTOTAL,CB,VB run=cs2agbibr6_800nm
