
#此项目为实验室异常动作的识别

from argparse import ArgumentParser
from typing import Dict
from mmpose.apis.inferencers import MMPoseInferencer, get_model_aliases
import sys
import time
from collections import defaultdict

filter_args = dict(bbox_thr=0.3, nms_thr=0.3, pose_based_nms=False)
POSE2D_SPECIFIC_ARGS = dict(
    yoloxpose=dict(bbox_thr=0.01, nms_thr=0.65, pose_based_nms=True),
    rtmo=dict(bbox_thr=0.1, nms_thr=0.65, pose_based_nms=True),
)


def parse_args():
    try:
        parser = ArgumentParser()
        parser.add_argument(
            'inputs',
            type=str,
            nargs='?',
            default=r'D:\PythonProject\mmpose\video\test414.mp4',
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
            default='cuda:0',
            help='Device used for inference. '
                 'If not specified, the available device will be automatically used.')
        parser.add_argument(
            '--show-progress',
            action='store_true',
            help='Display the progress bar during inference.')

        args, _ = parser.parse_known_args()

        for model in POSE2D_SPECIFIC_ARGS:
            if args.pose2d is not None and model in args.pose2d:
                filter_args.update(POSE2D_SPECIFIC_ARGS[model])
                break

        parser.add_argument(
            '--show',
            action='store_true',
            default=True,
            help='Display the image/video in a popup window.')
        parser.add_argument(
            '--draw-bbox',
            action='store_true',
            default=True,  # Set default to True
            help='Whether to draw the bounding boxes.')
        parser.add_argument(
            '--draw-heatmap',
            action='store_true',
            help='Whether to draw the predicted heatmaps.')
        parser.add_argument(
            '--bbox-thr',
            type=float,
            default=filter_args['bbox_thr'],
            help='Bounding box score threshold')
        parser.add_argument(
            '--nms-thr',
            type=float,
            default=filter_args['nms_thr'],
            help='IoU threshold for bounding box NMS')
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
            help='Plot predictions on a black image')
        parser.add_argument(
            '--vis-out-dir',
            type=str,
            help='Directory for saving visualized results.')
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

    except Exception as e:
        print(f"Error during argument parsing: {e}")
        sys.exit(1)


def display_model_aliases(model_aliases: Dict[str, str]) -> None:
    try:
        aliases = list(model_aliases.keys())
        max_alias_length = max(map(len, aliases))
        print(f'{"ALIAS".ljust(max_alias_length + 2)}MODEL_NAME')
        for alias in sorted(aliases):
            print(f'{alias.ljust(max_alias_length + 2)}{model_aliases[alias]}')
    except Exception as e:
        print(f"Error during model alias display: {e}")
        sys.exit(1)


def main():
    try:
        # 初始化计时器
        timers = defaultdict(float)
        total_start = time.time()

        init_args, call_args, display_alias = parse_args()
        timers['parse_args'] = time.time() - total_start

        if display_alias:
            model_aliases = get_model_aliases(init_args['scope'])
            display_model_aliases(model_aliases)
        else:
            # 模型初始化计时
            model_start = time.time()
            inferencer = MMPoseInferencer(**init_args)
            timers['model_init'] = time.time() - model_start

            frame_counter = 0
            total_frames = 0
            process_start = time.time()

            for result in inferencer(**call_args):
                frame_counter += 1
                total_frames += 1

                # 每2帧删除1帧，提高实时性
                if frame_counter % 2 == 0:
                    continue

                # 分析推理过程的各个步骤
                step_start = time.time()

                # 1. 检测框处理
                if hasattr(result, 'bboxes') and result.bboxes is not None:
                    bbox_time = time.time() - step_start
                    timers['bbox_process'] += bbox_time
                    step_start = time.time()
                    print(f"检测框处理时间: {bbox_time * 1000:.2f}ms")  # 添加详细日志

                # 2. 关键点检测
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    kpt_time = time.time() - step_start
                    timers['keypoint_detect'] += kpt_time
                    step_start = time.time()
                    print(f"关键点检测时间: {kpt_time * 1000:.2f}ms")  # 添加详细日志

                # 3. 骨骼点绘制
                if hasattr(result, 'visualization') and result.visualization is not None:
                    vis_time = time.time() - step_start
                    timers['skeleton_draw'] += vis_time
                    step_start = time.time()
                    print(f"骨骼点绘制时间: {vis_time * 1000:.2f}ms")  # 添加详细日志

                # 4. LSTM预测
                if hasattr(result, 'predictions') and result.predictions is not None:
                    lstm_time = time.time() - step_start
                    timers['lstm_predict'] += lstm_time
                    print(f"LSTM预测时间: {lstm_time * 1000:.2f}ms")  # 添加详细日志

                # 打印每帧的总处理时间
                if frame_counter % 30 == 0:  # 每30帧打印一次
                    frame_total_time = time.time() - step_start
                    print(f"\nFrame {frame_counter} 总处理时间: {frame_total_time * 1000:.2f}ms")

            # 计算平均时间
            processed_frames = frame_counter // 2  # 因为每2帧处理1帧
            if processed_frames > 0:
                # 计算每个步骤的平均时间
                for key in ['bbox_process', 'keypoint_detect', 'skeleton_draw', 'lstm_predict']:
                    if key in timers and timers[key] > 0:
                        timers[f'avg_{key}'] = timers[key] / processed_frames * 1000  # 转换为毫秒
                    else:
                        timers[f'avg_{key}'] = 0

                # 打印详细的性能分析结果
                print("\n推理过程性能分析:")
                print(f"总帧数: {total_frames}")
                print(f"处理帧数: {processed_frames}")
                print(f"参数解析时间: {timers['parse_args']:.2f}秒")
                print(f"模型初始化时间: {timers['model_init']:.2f}秒")

                # 计算总处理时间和FPS
                total_process_time = time.time() - process_start
                print(f"\n总处理时间: {total_process_time:.2f}秒")
                print(f"平均每帧总处理时间: {total_process_time / processed_frames * 1000:.2f}毫秒")
                print(f"处理速度: {processed_frames / total_process_time:.2f} FPS")

                # 打印每个步骤的总时间
                print("\n各步骤总耗时:")

            else:
                print("警告: 没有处理任何帧")

    except Exception as e:
        print(f"Error during inference: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
