# Cs2AgBiBr6 参考电池（复刻 Silvaco 输入卡 test0408，吸收层 800 nm）
#
# 参数来源：D:\王继凯毕设资料\原始数据\test0408_3D_TiO2_NiO_loop_perov.in
# 结构：FTO 20 / TiO2 20 / Cs2AgBiBr6 800 / NiO 20 / Au 100 nm
# 参考数据：D:\交互界面_data\datasets\jv_800nm.xlsx（Silvaco 导出）
#
# 已知差异（对照时需记住）：
#   * Atlas 用 index.file 的 n,k 做光学；本平台用内置 α(E) 表，Jsc 会有系统偏差；
#   * Atlas 开了 auger + optr 与 1e14 cm^-3 体陷阱、两个界面陷阱；
#     本平台当前只有 SRH（用 taun/taup 等效），复合会被低估；
#   * Atlas 的 gold 是 conductor；本平台按 .mat 的 AU 处理。
# 因此这个 deck 用于"同结构、同电学参数"的基线对照，不追求逐点吻合。

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
         mun=9.28 mup=9.28 taun=6.6e-7 taup=6.6e-7

material name=NIO eg=3.8 chi=1.46 eps=10.7 nc=2.8e19 nv=1e19 \
         mun=12 mup=2.8 taun=1e-6 taup=1e-5

models srh
optics spectrum=am15g intensity=100 bounces=1 bandgap.mask \
       nk.dir="D:/交互界面_data/datasets/nk"

method newton maxiter=80 tol=1e-8 damping=1.0
solve vstart=0 vfinal=2.0 vstep=0.02

output fields=X,PSI,N,P,EFIELD,JTOTAL,CB,VB run=cs2agbibr6_800nm
