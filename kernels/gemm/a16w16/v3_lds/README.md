# v3


lds layout without swizzling
```
python3 plot_layout.py lds --tensorShape 256 64 --kWidth 8 --nonKDim 16 --banks 64 --layout none --access read --swizzleVec 8 --output v3_lds_no-swizzling
```

lds layout with swizzling
```
python3 plot_layout.py lds --tensorShape 256 64 --kWidth 8 --nonKDim 16 --banks 64 --layout swizzle --access read --swizzleVec 8 --output v3_lds_swizzling
```

lds layout with padding
```
python3 plot_layout.py lds --tensorShape 256 64 --kWidth 8 --nonKDim 16 --banks 64 --layout padding --access read --swizzleVec 8 --sharedLayout "[[512, 16]], [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [16, 0],[32, 0], [64, 0], [1, 0], [2, 0], [4, 0], [8, 0], [128, 0]]" --dtype fp16 --output v3_lds_padding_512-16
```



fp8 with padding 1024:16
```
python3 plot_layout.py lds --tensorShape 256 128 --kWidth 32 --nonKDim 16 --banks 64 --layout padding --access read --swizzleVec 16 --sharedLayout "[[1024, 16]], [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [16, 0],[32, 0], [64, 0], [1, 0], [2, 0], [4, 0], [8, 0], [128, 0]]" --dtype fp8
```

fp8 with padding 1024:16,2048:32

```
python3 plot_layout.py lds --tensorShape 256 128 --kWidth 32 --nonKDim 16 --banks 64 --layout padding --access read --swizzleVec 16 --sharedLayout "[[1024, 16], [2048, 32]], [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [16, 0],[32, 0], [64, 0], [1, 0], [2, 0], [4, 0], [8, 0], [128, 0]]" --dtype fp8
```

