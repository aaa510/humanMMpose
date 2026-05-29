import json
import os
import numpy as np
from mmpose.apis import MMPoseInferencer

# 创建输出目录
output_dir = 'output_results'
vis_dir = os.path.join(output_dir, 'visualizations')
json_dir = os.path.join(output_dir, 'predictions')

os.makedirs(vis_dir, exist_ok=True)
os.makedirs(json_dir, exist_ok=True)

# 创建推理器
inferencer = MMPoseInferencer('human')

# 输入图像路径
img_path = 'tests/data/coco/000000000785.jpg'
img_name = os.path.splitext(os.path.basename(img_path))[0]

# 进行推理并保存可视化结果
result_generator = inferencer(
    img_path,
    show=True,
    vis_out_dir=vis_dir,
    radius=4,
    thickness=2
)

# 获取预测结果
result = next(result_generator)
predictions = result['predictions']

# 将NumPy数据类型转换为Python原生类型的函数
def convert_to_json_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    return obj

# 转换预测结果
predictions_converted = convert_to_json_serializable(predictions)

# 保存JSON文件
json_path = os.path.join(json_dir, f'{img_name}_prediction.json')
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(predictions_converted, f, ensure_ascii=False, indent=2)

print(f'可视化结果已保存至: {vis_dir}')
print(f'预测结果JSON已保存至: {json_path}')