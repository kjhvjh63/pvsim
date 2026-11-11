# MAPbI3 NIP 参考电池 —— 对应 param.schema.mapbi3_stack()
# 语法子集见 python/pvsim/script/parser.py 顶部说明
title MAPbI3 NIP reference cell

mesh nodes=101 interface.refine=1.0 space.mult=1.0

region num=1 material=FTO    thickness=200
region num=2 material=TiO2   thickness=40
region num=3 material=MAPbI3 thickness=500
region num=4 material=Spiro  thickness=200
region num=5 material=Au     thickness=80

# 电极掺杂（FTO n 型、Au p 型）
doping region=1 n.type conc=1e20
doping region=5 p.type conc=1e20

# 吸收层参数覆盖：带隙 1.6 eV、SRH 寿命 8 ns
material name=MAPBI3 eg=1.6 chi=3.93 eps=24 nc=2.2e18 nv=1e19 \
         mun=1.6 mup=1.7 taun=8e-9 taup=8e-9

models srh
optics spectrum=am15g intensity=100 bounces=1 bandgap.mask

method newton maxiter=60 tol=1e-8 damping=1.0
solve vstart=0 vfinal=1.8 vstep=0.05

log file=out/jv.log
output fields=X,PSI,N,P,EFIELD,JTOTAL,G_OPT,CB,VB run=mapbi3_ref
