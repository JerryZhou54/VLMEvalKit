import torch
import torch.distributed as dist
from vlmeval.config import supported_VLM
from vlmeval.utils import track_progress_rich
from vlmeval.smp import *
import json

FAIL_MSG = 'Failed to obtain answer via API.'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--model', type=str, nargs='+', required=True)
    parser.add_argument('--nproc', type=int, default=4, required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args

def infer_data(model, model_name, work_dir, json_dir, verbose=False, api_nproc=4):
    with open(json_dir, 'r') as f:
        text_output_dict = json.load(f)
    new_output_dict = {}
    model = supported_VLM[model_name]() if isinstance(model, str) else model
    for k, v_dict in tqdm(text_output_dict.items()):
        response_dict = model.generate(text_output_dict=v_dict)
        torch.cuda.empty_cache()

        if verbose:
            print(response_dict, flush=True)

        new_output_dict[k] = response_dict
    
    base, ext = os.path.splitext(json_dir)
    new_json_dir = f"{base}_new{ext}"
    with open(new_json_dir, "w") as f:
        json.dump(new_output_dict, f, indent=4)

    print(f"Finish writing output to {new_json_dir}")

    return model


# A wrapper for infer_data, do the pre & post processing
def infer_data_job(model, work_dir, model_name, json_dir, verbose=False, api_nproc=4, ignore_failed=False):
    rank, world_size = get_rank_and_world_size()

    model = infer_data(
        model=model, work_dir=work_dir, model_name=model_name, json_dir=json_dir,
        verbose=verbose, api_nproc=api_nproc)
    if world_size > 1:
        dist.barrier()
    return model
