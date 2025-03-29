import json

import torch
import torch.distributed as dist

from vlmeval.config import supported_VLM
from vlmeval.dataset.video_dataset_config import supported_video_datasets
from vlmeval.dataset import build_dataset
from vlmeval.inference import infer_data_job
from vlmeval.inference_video import infer_data_job_video
from vlmeval.inference_mt import infer_data_job_mt
from vlmeval.smp import *
from vlmeval.utils.result_transfer import MMMU_result_transfer, MMTBench_result_transfer

def build_model_from_config(cfg, model_name):
    import vlmeval.vlm
    config = cp.deepcopy(cfg[model_name])
    if config == {}:
        return supported_VLM[model_name]()
    assert 'class' in config
    cls_name = config.pop('class')
    if hasattr(vlmeval.api, cls_name):
        return getattr(vlmeval.api, cls_name)(**config)
    elif hasattr(vlmeval.vlm, cls_name):
        return getattr(vlmeval.vlm, cls_name)(**config)
    else:
        raise ValueError(f'Class {cls_name} is not supported in `vlmeval.api` or `vlmeval.vlm`')

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    # Essential Args, Setting the Names of Datasets and Models
    parser.add_argument('--json_dir', type=str, help='Dir of output json file')
    parser.add_argument('--model', type=str, nargs='+', help='Names of Models')
    # Work Dir
    parser.add_argument('--work-dir', type=str, default='./outputs', help='select the output directory')
    # Logging Utils
    parser.add_argument('--verbose', action='store_true')
    # Configuration for Resume
    # Ignore: will not rerun failed VLM inference
    parser.add_argument('--ignore', action='store_true', help='Ignore failed indices. ')
    # Reuse: will reuse the existing prediction files
    parser.add_argument('--reuse', action='store_true')
    # Reuse-aux: if set, when reuse is True, will also reuse the auxiliary evaluation files
    parser.add_argument('--reuse-aux', type=bool, default=True, help='reuse auxiliary evaluation files')

    args = parser.parse_args()
    return args


def main():
    logger = get_logger('RUN')
    rank, world_size = get_rank_and_world_size()
    args = parse_args()
    use_config, cfg = False, None
    assert args.json_dir, '--json_dir should be the dir to the output json file'

    if rank == 0:
        if not args.reuse:
            logger.warning('--reuse is not set, will not reuse previous (before one day) temporary files')
        else:
            logger.warning('--reuse is set, will reuse the latest prediction & temporary pickle files')

    if 'MMEVAL_ROOT' in os.environ:
        args.work_dir = os.environ['MMEVAL_ROOT']

    for k, v in supported_VLM.items():
        if hasattr(v, 'keywords') and 'retry' in v.keywords and args.retry is not None:
            v.keywords['retry'] = args.retry
            supported_VLM[k] = v
        if hasattr(v, 'keywords') and 'verbose' in v.keywords and args.verbose is not None:
            v.keywords['verbose'] = args.verbose
            supported_VLM[k] = v

    if world_size > 1:
        local_rank = os.environ.get('LOCAL_RANK', 0)
        torch.cuda.set_device(int(local_rank))
        dist.init_process_group(
            backend='nccl',
            timeout=datetime.timedelta(seconds=int(os.environ.get('DIST_TIMEOUT', 3600)))
        )

    for _, model_name in enumerate(args.model):
        model = None
        date, commit_id = timestr('day'), githash(digits=8)
        eval_id = f"T{date}_G{commit_id}"

        pred_root = osp.join(args.work_dir, model_name, eval_id)
        pred_root_meta = osp.join(args.work_dir, model_name)
        os.makedirs(pred_root_meta, exist_ok=True)

        prev_pred_roots = ls(osp.join(args.work_dir, model_name), mode='dir')
        if len(prev_pred_roots) and args.reuse:
            prev_pred_roots.sort()

        if not osp.exists(pred_root):
            os.makedirs(pred_root, exist_ok=True)

        if use_config:
            model = build_model_from_config(cfg['model'], model_name)

        if world_size > 1:
            dist.barrier()        

        if model is None:
            model = model_name  # which is only a name

        # Perform the Inference
        model = infer_data_job(
            model,
            work_dir=pred_root,
            model_name=model_name,
            json_dir=args.json_dir,
            verbose=args.verbose,
            api_nproc=args.api_nproc,
            ignore_failed=args.ignore)

        if world_size > 1:
            dist.barrier()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    load_env()
    main()
