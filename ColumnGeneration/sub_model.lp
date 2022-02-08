\ Model sub model
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  0.5 y[1] + 0.5 y[2] + y[3]
Subject To
 R0: 3 y[0] + 7 y[1] + 9 y[2] + 16 y[3] <= 20
Bounds
Generals
 y[0] y[1] y[2] y[3]
End
