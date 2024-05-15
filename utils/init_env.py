# 获取 路径
import os
import sys

file_path = os.path.dirname(os.path.abspath(__file__))
# 修改运行路径
sys.path.append(file_path)
# 0 表示优先级，数字越大级别越低，修改模块的导入
sys.path.insert(0, os.path.dirname(file_path))