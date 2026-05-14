import os
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")


class WISDM(Dataset):
    def __init__(
        self,
        root_path,
        flag="train",
        seq_len=200,
        data_path="WISDM.txt",
        scale=True,
        drop_short=False,
    ):
        """
        seq_len: seq_len
        flag: 'train', 'val', or 'test'
        """
        self.label_map = {
            "Walking": 0,
            "Jogging": 1,
            "Sitting": 2,
            "Standing": 3,
            "Upstairs": 4,
            "Downstairs": 5,
        }
        self.labels = []

        self.seq_len = seq_len  # 200

        # Validation
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.flag = flag
        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path

        # Store lists to hold the processed sequences
        self.data_x_seq = []
        self.data_y = []
        self.data_stamp_seq = []

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        # 1. Load Data
        COLUMN_NAMES = ["user", "activity", "timestamp", "x-axis", "y-axis", "z-axis"]
        df_raw = pd.read_csv(
            os.path.join(self.root_path, self.data_path),
            header=None,
            names=COLUMN_NAMES,
            on_bad_lines="skip",
        )

        # 2. Clean Data
        df_raw["z-axis"] = (
            df_raw["z-axis"].astype(str).str.replace(";", "").astype(float)
        )
        df_raw = df_raw.dropna()

        # remove all line that have 1 or more 0 in the colummns x-axis, y-axis or z-axis
        df_raw = df_raw[
            (df_raw[["timestamp", "x-axis", "y-axis", "z-axis"]] != 0).all(axis=1)
        ]

        # convert all activity by their encoded values
        df_raw["encoded_activity"] = df_raw["activity"].map(self.label_map)

        # Convert timestamp to datetime for embedding generation
        # WISDM timestamps are usually unix epoch in milliseconds
        df_raw["datetime"] = pd.to_datetime(df_raw["timestamp"], unit="ns")

        # 3. Global Scaling (Optional)
        # We fit the scaler on the raw values first, regardless of grouping
        if self.scale:
            raw_values = df_raw[["x-axis", "y-axis", "z-axis"]].values
            self.scaler.fit(raw_values)

        # 4. Group by User and Activity to ensure continuity
        groups = df_raw.groupby(["user", "activity"])

        # data_name = self.data_path.split(".")[0]
        # pt_file_path = os.path.join("./dataset", f"{data_name}.pt")
        # preprocess_data = torch.load(pt_file_path)

        for (user, activity), group in groups:
            # Sort by timestamp to ensure time order
            group = group.sort_values("timestamp")

            values = group[["x-axis", "y-axis", "z-axis"]].values
            classes = group[["encoded_activity"]].values

            # Scale this group's data if enabled
            if self.scale:
                values = self.scaler.transform(values)

            num_points = len(values)

            if num_points < self.seq_len:
                print(
                    "not enough points in this series, (", num_points, "), skipping it"
                )
                break

            activity_scaling = {
                "Walking": 4,
                "Jogging": 4,
                "Sitting": 4,
                "Standing": 4,
                "Upstairs": 4,
                "Downstairs": 4,
            }
            for i in range(
                0,
                num_points - self.seq_len - 1,
                self.seq_len // activity_scaling[activity],
            ):
                end = i + self.seq_len
                seq_x = values[i:end]
                y = classes[end - 1]

                # seq_x_mark = preprocess_data[i:end]
                # y_mark = preprocess_data[end]

                # Store
                # self.labels.append(y_mark)
                self.data_x_seq.append(seq_x)
                self.data_y.append(y)
                # self.data_stamp_seq.append((seq_x_mark, y_mark))

        # 5. Split into Train/Val/Test
        num_total = len(self.data_x_seq)
        print("nbr of timeseries :", num_total)
        num_train = int(num_total * 0.7)
        num_test = int(num_total * 0.2)
        num_val = num_total - num_train - num_test

        # TODO: do some every day im shuffling (LMFAO)
        if self.set_type == 0:  # Train
            self.data_x_seq = self.data_x_seq[:num_train]
            self.data_y = self.data_y[:num_train]
            # self.data_stamp_seq = self.data_stamp_seq[:num_train]
        elif self.set_type == 1:  # Val
            self.data_x_seq = self.data_x_seq[num_train : num_train + num_val]
            self.data_y = self.data_y[num_train : num_train + num_val]
            # self.data_stamp_seq = self.data_stamp_seq[num_train : num_train + num_val]
        else:  # Test
            self.data_x_seq = self.data_x_seq[num_train + num_val :]
            self.data_y = self.data_y[num_train + num_val :]
            # self.data_stamp_seq = self.data_stamp_seq[num_train + num_val :]

        # 6. Print class distribution for each set (data_y)
        unique, counts = np.unique(self.data_y, return_counts=True)
        print("Class distribution:", dict(zip(unique, counts)))

        print(f"Dataset {self.flag} created. Total sequences: {len(self.data_x_seq)}")

    def __getitem__(self, index):
        seq_x = self.data_x_seq[index]
        y = self.data_y[index]
        return (
            torch.tensor(seq_x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(seq_x, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.data_x_seq)


class Dataset_Preprocess(Dataset):
    def __init__(
        self,
        root_path,
        flag="train",
        seq_len=None,
        data_path="",
        scale=True,
        seasonal_patterns=None,
    ):
        self.seq_len = seq_len
        self.flag = flag
        self.root_path = root_path
        self.data_path = data_path
        self.label_map = {
            "Walking": 0,
            "Jogging": 1,
            "Sitting": 2,
            "Standing": 3,
            "Upstairs": 4,
            "Downstairs": 5,
        }

        COLUMN_NAMES = ["user", "activity", "timestamp", "x-axis", "y-axis", "z-axis"]
        df_raw = pd.read_csv(
            os.path.join(self.root_path, self.data_path),
            header=None,
            names=COLUMN_NAMES,
            on_bad_lines="skip",
        )

        # 2. Clean Data
        df_raw["z-axis"] = (
            df_raw["z-axis"].astype(str).str.replace(";", "").astype(float)
        )
        self.df_raw = df_raw.dropna()

        # remove all line that have 1 or more 0 in the colummns x-axis, y-axis or z-axis
        print(df_raw.shape)
        df_raw = df_raw[
            (df_raw[["timestamp", "x-axis", "y-axis", "z-axis"]] != 0).all(axis=1)
        ]
        print(df_raw.shape)

    def __getitem__(self, index):
        # TODO : remove the 10secs in hard text
        seq_x_mark = f"This is Time Series of an activity among {self.label_map} of {self.seq_len} points over 10 seconds"
        return seq_x_mark

    def __len__(self):
        return len(self.df_raw)
