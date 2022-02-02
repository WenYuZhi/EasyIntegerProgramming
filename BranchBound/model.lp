\ Model mip1_copy_copy_copy_copy_copy
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  x + y + 2 z
Subject To
 c0: x + 2 y + 3 z <= 3.9
 c1: x + y >= 1
 R2: x = 1
 R3: y = 1
Bounds
 x <= 1
 y <= 1
 z <= 1
End
