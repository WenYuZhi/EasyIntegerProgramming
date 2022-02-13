\ Model subproblem
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  - 500 u[0] + 100 u[1] + 100 u[2] + 100 u[3] + 100 u[4] + 100 u[5]
   + 100 u[6] + 100 u[7] + 100 u[8] + 100 u[9] + 100 u[10]
Subject To
 R0: u[0] + u[1] >= 1
 R1: u[0] + u[2] >= 1.01
 R2: u[0] + u[3] >= 1.02
 R3: u[0] + u[4] >= 1.03
 R4: u[0] + u[5] >= 1.04
 R5: u[0] + u[6] >= 1.05
 R6: u[0] + u[7] >= 1.06
 R7: u[0] + u[8] >= 1.07
 R8: u[0] + u[9] >= 1.08
 R9: u[0] + u[10] >= 1.09
Bounds
End
