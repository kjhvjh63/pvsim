# Atlas / DeckBuild 风格示例 —— 与本平台语句混用均可
#
# 演示对齐 Silvaco 的写法：
#   * mesh nx= 代替 nodes=
#   * region 用 x.min/x.max（单位 µm）链式定义层栈
#   * doping uniform n.type / p.type
#   * electrode 代替 contact
#   * output 用裸标志（con.band val.band j.total ...）
#   * save / extract / tonyplot / go / quit 等流程语句原样保留
#
# 结构等价于 device_example.tst 的 5 层 NIP 电池（FTO/TiO2/MAPbI3/Spiro/Au）。

title Atlas-style MAPbI3 NIP cell

mesh nx=101 interface.refine=1.0 space.mult=1.0 temperature=300

region num=1 material=FTO    x.min=0    x.max=0.2
region num=2 material=TiO2   x.max=0.24
region num=3 material=MAPbI3 x.max=0.74
region num=4 material=Spiro  x.max=0.94
region num=5 material=Au     x.max=1.02

doping region=1 uniform n.type conc=1e20
doping region=5 uniform p.type conc=1e20

material name=MAPBI3 eg=1.6 chi=3.93 eps=24 nc=2.2e18 nv=1e19 \
         mun=1.6 mup=1.7 taun=8e-9 taup=8e-9

models srh
optics spectrum=am15g intensity=100 bounces=1 bandgap.mask

method newton maxiter=60 tol=1e-8 damping=1.0
solve vanode=0 vfinal=1.3 vstep=0.05

output con.band val.band e.field j.total qfn qfp run=atlas_style

# 以下语句解析但不产生计算副作用
save outfile=out/atlas_style.str
extract name="voc" curve="jv"
tonyplot file=out/atlas_style.log
go
quit
