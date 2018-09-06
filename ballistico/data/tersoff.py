
tersoff = {
   'potentials': [
      {
         'name': 'C',
         # 'params': 'ters      1544.8      3.4653      389.63      2.3064        1.80\n        2.10   4.1612e-6     0.99054     19981.0      7.0340    -0.39953\n',
         # 'pa': 1544.8,
         # 'pl1': 3.4653,
         # 'pb': 389.63,
         # 'pl2': 2.3064,
         # 'pr': 1.80,
         # 'ps': 2.10,
         # 'pbt': 4.1612e-6,
         # 'pn': 0.99054,
         # 'pc': 19981.0,
         # 'pd': 7.0340,
         # 'ph': -0.39953
   
         'pa': 1393.6,
         'pl1': 3.4879,
         'pb': 430.00,
         'pl2': 2.2119,
         'pr': 1.80,
         'ps': 2.10,
         'pbt': 1.5724e-7,
         'pn': 0.72751,
         'pc': 38049.0,
         'pd': 4.3484,
         'ph': -0.930
      },
      {
         'name': 'Si',
         # 'params': 'ters      1830.8      2.4799      471.18      1.7322        2.70\n        3.00   1.1000e-6     0.78734    100390.0      16.217    -0.59825\n',
         #                     'pa'       'pl1'        'pb'       'pl2'          'pr\n'       'ps'       'pbt'      'pn'       'pc'          'pd'        'ph'
         'pa': 1830.8,
         'pl1': 2.4799,
         'pb': 471.18,
         'pl2': 1.7322,
         'pr': 2.70,
         'ps': 3.00,
         'pbt': 1.1000e-6,
         'pn': 0.78734,
         'pc': 100390.0,
         'pd': 16.217,
         'ph': -0.59825
      },
      {
         'name': 'Ge',
         'pa' : 1769.0,
         'pb' : 419.23,
         'pl1' : 2.4451,
         'pl2' : 1.7047,
         'pr' : 2.8,
         'pn' : .75627,
         'pbt' : (9.0166e-7),
         'pc' : (106430.0),
         'pd' : (15.652),
         'ph' : -.43884,
         'ps' : 3.1
      },
      {
         'name': 'Ga',
         'pa' : 993.888094,
         'pb' : 136.123032,
         'pl1' : 2.50842747,
         'pl2' : 1.490824,
         'pn' : 3.4729041,
         'pbt' : (0.23586237),
         'pc' : (0.07629773),
         'pd' : (19.796474),
         'ph' : 7.1459174,
         'pr' : 3.4,
         'ps' : 3.6
      },
      {
         'name': 'As',
         'pa' : 1571.86084,
         'pb' : 546.4316579,
         'pl1' : 2.384132239,
         'pl2' : 1.7287263,
         'pn' : 0.60879133,
         'pbt' : (0.00748809),
         'pc' : (5.273131),
         'pd' : (0.75102662),
         'ph' : 0.15292354,
         'pr' : 3.4,
         'ps' : 3.6
      }
   ],
   'cross_terms': [
      {
         # TODO: This parameter is taken from Yip, which is different from dlpoly examples and phonts
         'name_1': 'Si',
         'name_2': 'C',
         # Atomistic modeling of finite temperature properties of 3-Sic. 1 Lattice vibrations, heat capacity, and thermal expansion (1-7 - 2002)
         'params': '1.0086 1.0\n'
         
         # Dlpoly / PhonTS
         # 'params': '0.9776 1.0\n'
      },
      {
         'name_1': 'Si',
         'name_2': 'Ge',
         'params': '1.00061 1.00061\n'
      },
      {
         'name_1': 'C',
         'name_2': 'Ge',
         'params': '1.0 1.0\n'
      }
   ]
}



#
#
#       do i = 1,ntmx
#         if (name(i).eq."Cx".or.name(i).eq."C") then
# !   corrected ones are from PhysRevB.81.205441 (pb=430.0, ph=-0.93)
#
#           pa(i)   = 1393.6d0
#           pb(i)   = 346.74d0
#           pl1(i)  = 3.4879d0
#           pl2(i)  = 2.2119d0
#           pn(i)   = .72751d0
#           pbt(i)  = (1.5724d-7)**pn(i)
#           pc(i)   = (38049.0d0)**2
#           pd(i)   = (4.3484d0)**2
#           ph(i)   = -.57058d0
#           pr(i)   = 1.8d0
#           ps(i)   = 2.1d0
#           c2d2(i) = 1.0d0+pc(i)/pd(i)
#           pwr(i)  = -1.0d0/(2.0d0*pn(i))
#           pwr1(i) = pwr(i)-1.0d0
#           pwr2(i) = pn(i)-1.0d0
#           chij(i,i) = 1.0
#
#         endif
#
#         if (name(i).eq."Si") then
#           pa(i)   = 1830.8d0
#           pb(i)   = 471.18d0
#           pl1(i)  = 2.4799d0
#           pl2(i)  = 1.7322d0
#           pn(i)   = .78734d0
#           pbt(i)  = (1.1d-6)**pn(i)
#           pc(i)   = (100390.0d0)**2
#           pd(i)   = (16.217d0)**2
#           ph(i)   = -.59825d0
#           pr(i)   = 2.7d0
#           ps(i)   = 3.0d0
#           c2d2(i) = 1.0d0+pc(1)/pd(i)
#           pwr(i)  = -1.0d0/(2.0d0*pn(i))
#           pwr1(i) = pwr(i)-1.0d0
#           pwr2(i) = pn(i)-1.0d0
#           chij(i,i) = 1.0
#         endif
#
#         if (name(i).eq."Ge") then
#           pa(i)   = 1769.0d0
#           pb(i)   = 419.23d0
#           pl1(i)  = 2.4451d0
#           pl2(i)  = 1.7047d0
#           pn(i)   = .75627d0
#           pbt(i)  = (9.0166d-7)**pn(i)
#           pc(i)   = (106430.0d0)**2
#           pd(i)   = (15.652d0)**2
#           ph(i)   = -.43884d0
#           pr(i)   = 2.8d0
#           ps(i)   = 3.1d0
#           c2d2(i) = 1.0d0+pc(i)/pd(i)
#           pwr(i)  = -1.0d0/(2.0d0*pn(i))
#           pwr1(i) = pwr(i)-1.0d0
#           pwr2(i) = pn(i)-1.0d0
#           chij(i,i) = 1.0
#         endif
#
#         if (name(i).eq."Ga") then
#           pa(i)   = 993.888094d0
#           pb(i)   = 136.123032d0
#           pl1(i)  = 2.50842747d0
#           pl2(i)  = 1.490824d0
#           pn(i)   = 3.4729041
#           pbt(i)  = (0.23586237)**pn(i)
#           pc(i)   = (0.07629773)**2
#           pd(i)   = (19.796474)**2
#           ph(i)   = 7.1459174d0
#           pr(i)   = 3.4d0
#           ps(i)   = 3.6d0
#           c2d2(i) = 1.0d0+pc(i)/pd(i)
#           pwr(i)  = -1.0d0/(2.0d0*pn(i))
#           pwr1(i) = pwr(i)-1.0d0
#           pwr2(i) = pn(i)-1.0d0
#           chij(i,i) = 1.0
#         endif
#
#         if (name(i).eq."As") then
#           pa(i)   = 1571.86084d0
#           pb(i)   = 546.4316579d0
#           pl1(i)  = 2.384132239d0
#           pl2(i)  = 1.7287263d0
#           pn(i)   = 0.60879133d0
#           pbt(i)  = (0.00748809d0)**pn(i)
#           pc(i)   = (5.273131d0)**2
#           pd(i)   = (0.75102662d0)**2
#           ph(i)   = 0.15292354d0
#           pr(i)   = 3.4d0
#           ps(i)   = 3.6d0
#           c2d2(i) = 1.0d0+pc(i)/pd(i)
#           pwr(i)  = -1.0d0/(2.0d0*pn(i))
#           pwr1(i) = pwr(i)-1.0d0
#           pwr2(i) = pn(i)-1.0d0
#           chij(i,i) = 1.0
#         endif
#
#
#         do j = i+1,ntmx
#           if ((name(i).eq."Si".and.name(j).eq."Ge").or.(name(j).eq."Si".and.name(i).eq."Ge")) then
#             chij(i,j) = 1.00061
#             chij(j,i) = chij(i,j)
#           endif
#           if (((name(i).eq."Si".and.name(j).eq."Cx").or.(name(j).eq."Si".and.name(i).eq."Cx")).or.&
#               ((name(i).eq."Si".and.name(j).eq."C").or.(name(j).eq."Si".and.name(i).eq."C"))) then
#             chij(i,j) = 0.9776
#             chij(j,i) = chij(i,j)
#           endif
#           if (((name(i).eq."Cx".and.name(j).eq."Ge").or.(name(j).eq."Cx".and.name(i).eq."Ge")).or.&
#               ((name(i).eq."C".and.name(j).eq."Ge").or.(name(j).eq."C".and.name(i).eq."Ge")))then
#             chij(i,j) = 1.0
#             chij(j,i) = chij(i,j)
#           endif
#         enddo
#       enddo
# !