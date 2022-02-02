\ Model mip1
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  x[0] + x[1] + 2 x[2] + 2 x[8] + x[9]
Subject To
 c0: x[0] + 2 x[1] + 3 x[2] + 5 x[3] + 3 x[4] <= 8
 c1: 2 x[3] + 2 x[4] + 3 x[5] + 5 x[6] + 3 x[7] <= 10
 c2: x[7] + x[8] + 3 x[9] <= 4
 c3: 2 x[0] + x[2] + 3 x[7] + 3 x[8] + 2 x[9] <= 8
 c4: x[7] + x[8] + 3 x[9] >= 1
 c5: 2 x[4] + 2 x[5] + x[6] + 5 x[7] + 3 x[8] >= 4
Bounds
Binaries
 x[0] x[1] x[2] x[3] x[4] x[5] x[6] x[7] x[8] x[9]
End
