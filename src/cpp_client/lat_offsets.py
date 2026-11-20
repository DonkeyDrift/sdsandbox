#!/usr/bin/env python3
"""打印 fork send_reset_car 的 lateral_offset 随机化序列:
random.Random((seed, "lat")).uniform(-lat, lat) 四舍五入 3 位。
C++ 客户端启动时 popen 调用本脚本,避免嵌套引号转义地狱。

用法: python3 lat_offsets.py <seed_base> <count> [lat]
"""
import random
import sys

seed_base = int(sys.argv[1])
count = int(sys.argv[2])
lat = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
for s in range(seed_base, seed_base + count):
    print(round(random.Random((s, "lat")).uniform(-lat, lat), 3))
