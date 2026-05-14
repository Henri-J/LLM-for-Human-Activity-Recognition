import argparse

import torch
from torch.utils.data import DataLoader

from data_provider.data_loader import WISDM, Dataset_Preprocess
from models.Preprocess_Llama import Model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoTimes Preprocess")
    parser.add_argument("--gpu", type=int, default=0, help="gpu id")

    parser.add_argument(
        "--dataset",
        type=str,
        default="WISDM",
        help="dataset to preprocess, options:[WISDM]",
    )
    args = parser.parse_args()

    model = Model(args)

    assert args.dataset in ["WISDM"]
    if args.dataset == "WISDM":
        data_set = Dataset_Preprocess(
            root_path="./dataset/WISDM/",
            data_path="WISDM-100000.txt",
            seq_len=200,
        )

    data_loader = DataLoader(
        data_set,
        batch_size=128,
        shuffle=False,
    )

    from tqdm import tqdm

    save_dir_path = "./dataset/"
    output_list = []
    for idx, data in tqdm(enumerate(data_loader)):
        output = model(data)
        output_list.append(output.detach().cpu())
    result = torch.cat(output_list, dim=0)
    torch.save(result, save_dir_path + f"/{args.dataset}.pt")
