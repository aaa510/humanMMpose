import json
import os
from mmpose.apis import MMPoseInferencer

img_path = 'tests/data/coco/000000000785.jpg'  # 将 img_path 替换为你自己的路径

# 使用模型别名创建推理器
inferencer = MMPoseInferencer('human')

# MMPoseInferencer采用了惰性推断方法，在给定输入时创建一个预测生成器
result_generator = inferencer(img_path, show=True)
result = next(result_generator)

# 设置保存的文件夹
output_dir = 'predictions'
os.makedirs(output_dir, exist_ok=True)

# 构造保存结果的文件路径
result_filename = os.path.join(output_dir, 'result.json')

# 将结果保存为 JSON 文件
with open(result_filename, 'w') as f:
    json.dump(result, f, indent=4)

print(f'Results saved to {result_filename}')
