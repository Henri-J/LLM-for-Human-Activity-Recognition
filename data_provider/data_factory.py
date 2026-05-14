from torch.utils.data import DataLoader

from data_provider.data_loader import WISDM


def data_provider(args, flag):
    Data = WISDM

    if flag == "test":
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    elif flag == "val":
        shuffle_flag = args.val_set_shuffle
        drop_last = False
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = args.drop_last
        batch_size = args.batch_size

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        seq_len=args.seq_len,
        drop_short=args.drop_short,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
    )
    return data_set, data_loader
