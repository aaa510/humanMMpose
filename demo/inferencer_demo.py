from argparse import ArgumentParser
from typing import Dict
from mmpose.apis.inferencers import MMPoseInferencer, get_model_aliases
from AR.extract_points import extract_keypoints, save_to_excel
from AR.detection.det_lstm import detect_actions
from decision.dynamic_threat import ThreatAssessment, process_features  # ✅ 导入动态威胁
from decision.static_threat import compute_static_threat  # ✅ 导入静态威胁计算
from mmengine.structures import InstanceData  # ✅ 直接导入 InstanceData


threat_model = ThreatAssessment()


filter_args = dict(bbox_thr=0.3, nms_thr=0.3, pose_based_nms=False)
POSE2D_SPECIFIC_ARGS = dict(
    yoloxpose=dict(bbox_thr=0.01, nms_thr=0.65, pose_based_nms=True),
    rtmo=dict(bbox_thr=0.1, nms_thr=0.65, pose_based_nms=True),
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'inputs',
        type=str,
        nargs='?',
        default=r'C:\PythonProject\mmpose\mmpose-zhr\mmpose\test.mp4',
        help='Input image/video path or folder path.')

    # init args

    parser.add_argument(
        '--pose2d',
        type=str,
        default='human',  # Set default to 'human'
        help='Pretrained 2D pose estimation algorithm. It\'s the path to the '
             'config file or the model name defined in metafile.')
    parser.add_argument(
        '--pose2d-weights',
        type=str,
        default=None,
        help='Path to the custom checkpoint file of the selected pose model. '
             'If it is not specified and "pose2d" is a model name of metafile, '
             'the weights will be loaded from metafile.')
    parser.add_argument(
        '--pose3d',
        type=str,
        default=None,
        help='Pretrained 3D pose estimation algorithm. It\'s the path to the '
             'config file or the model name defined in metafile.')

    parser.add_argument(
        '--pose3d-weights',
        type=str,
        default=None,
        help='Path to the custom checkpoint file of the selected pose model. '
             'If it is not specified and "pose3d" is a model name of metafile, '
             'the weights will be loaded from metafile.')

    # 在default_det_models.py中定义了pose2d的模型和权重
    parser.add_argument(
        '--det-model',
        type=str,
        default=None,
        help='Config path or alias of detection model.')
    parser.add_argument(
        '--det-weights',
        type=str,
        default=None,
        help='Path to the checkpoints of detection model.')
    parser.add_argument(
        '--det-cat-ids',
        type=int,
        nargs='+',
        default=0,
        help='Category id for detection model.')
    parser.add_argument(
        '--scope',
        type=str,
        default='mmpose',
        help='Scope where modules are defined.')
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device used for inference. '
             'If not specified, the available device will be automatically used.')
    # 显示推理进度
    parser.add_argument(
        '--show-progress',
        action='store_true',
        help='Display the progress bar during inference.')

    # The default arguments for prediction filtering differ for top-down
    # and bottom-up models. We assign the default arguments according to the
    # selected pose2d model

    # 检查是否选用了特定的 2D 模型，若选用则更新默认的过滤参数。
    args, _ = parser.parse_known_args()
    for model in POSE2D_SPECIFIC_ARGS:
        if args.pose2d is not None and model in args.pose2d:
            filter_args.update(POSE2D_SPECIFIC_ARGS[model])
            break

    # call args
    # 返回解析结果
    parser.add_argument(
        '--show',
        action='store_true',
        default=True,
        help='Display the image/video in a popup window.')
    # 边界框
    parser.add_argument(
        '--draw-bbox',
        action='store_true',
        default=True,  # Set default to True
        help='Whether to draw the bounding boxes.')
    # 关键点热图
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        # default=True,
        help='Whether to draw the predicted heatmaps.')

    # 边界框分数的阈值，低于该阈值的边界框将被忽略。
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=filter_args['bbox_thr'],
        help='Bounding box score threshold')

    # 用于非极大值抑制（NMS）的 IoU 阈值。
    parser.add_argument(
        '--nms-thr',
        type=float,
        default=filter_args['nms_thr'],
        help='IoU threshold for bounding box NMS')

    # 是否使用基于姿态的 NMS。
    parser.add_argument(
        '--pose-based-nms',
        type=lambda arg: arg.lower() in ('true', 'yes', 't', 'y', '1'),
        default=filter_args['pose_based_nms'],
        help='Whether to use pose-based NMS')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--tracking-thr', type=float, default=0.3, help='Tracking threshold')
    parser.add_argument(
        '--use-oks-tracking',
        action='store_true',
        help='Whether to use OKS as similarity in tracking')
    parser.add_argument(
        '--disable-norm-pose-2d',
        action='store_true',
        help='Whether to scale the bbox (along with the 2D pose) to the '
             'average bbox scale of the dataset, and move the bbox (along with the '
             '2D pose) to the average bbox center of the dataset. This is useful '
             'when bbox is small, especially in multi-person scenarios.')
    parser.add_argument(
        '--disable-rebase-keypoint',
        action='store_true',
        default=False,
        help='Whether to disable rebasing the predicted 3D pose so its '
             'lowest keypoint has a height of 0 (landing on the ground). Rebase '
             'is useful for visualization when the model do not predict the '
             'global position of the 3D pose.')
    parser.add_argument(
        '--num-instances',
        type=int,
        default=1,
        help='The number of 3D poses to be visualized in every frame. If '
             'less than 0, it will be set to the number of pose results in the '
             'first frame.')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization.')
    parser.add_argument(
        '--thickness',
        type=int,
        default=2,
        help='Link thickness for visualization.')
    parser.add_argument(
        '--skeleton-style',
        default='mmpose',
        type=str,
        choices=['mmpose', 'openpose'],
        help='Skeleton style selection')
    parser.add_argument(
        '--black-background',
        action='store_true',
        # default=True,
        help='Plot predictions on a black image')
    parser.add_argument(
        '--vis-out-dir',
        type=str,
        # default='D:\Zher\Codes\mmpose/vis_results/rtm',
        help='Directory for saving visualized results.')
    # 数据推理结果保存路径
    parser.add_argument(
        '--pred-out-dir',
        type=str,
        default='',
        help='Directory for saving inference results.')
    parser.add_argument(
        '--show-alias',
        action='store_true',
        help='Display all the available model aliases.')

    call_args = vars(parser.parse_args())

    # 定义初始化需要的参数。
    init_kws = [
        'pose2d', 'pose2d_weights', 'scope', 'device', 'det_model',
        'det_weights', 'det_cat_ids', 'pose3d', 'pose3d_weights',
        'show_progress'
    ]
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)

    display_alias = call_args.pop('show_alias')

    return init_args, call_args, display_alias


def display_model_aliases(model_aliases: Dict[str, str]) -> None:
    """Display the available model aliases and their corresponding model
    names."""
    aliases = list(model_aliases.keys())
    max_alias_length = max(map(len, aliases))
    print(f'{"ALIAS".ljust(max_alias_length + 2)}MODEL_NAME')
    for alias in sorted(aliases):
        print(f'{alias.ljust(max_alias_length + 2)}{model_aliases[alias]}')



def main():
    init_args, call_args, display_alias = parse_args()
    if display_alias:
        model_alises = get_model_aliases(init_args['scope'])
        display_model_aliases(model_alises)
    else:
        inferencer = MMPoseInferencer(**init_args)
        count = 0
        stats_dict = {}

        keypoint_list = []  # 存储所有帧的关键点数据
        prev_frame_data = None  # 记录上一帧的关键点数据
        max_columns = 34  # **初始最大列数（单人）**

        # 进行推理，并提取每帧的关键点数据
        for result in inferencer(call_args['inputs']):
            predictions = result.get('predictions', [])  # 获取每帧的预测结果
            frame_data, prev_frame_data, max_columns = extract_keypoints(predictions, prev_frame_data,
                                                                         max_columns)  # 提取关键点
            keypoint_list.append(frame_data)  # 记录数据

        # 保存数据
        save_to_excel(keypoint_list, max_columns)



if __name__ == '__main__':
    main()

