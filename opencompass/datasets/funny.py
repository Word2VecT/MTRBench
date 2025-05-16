import datasets
import os.path as osp
import json

from datasets import Dataset, DatasetDict, concatenate_datasets

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset

@LOAD_DATASET.register_module()
class FunnyDataset(BaseDataset):

    @staticmethod
    def load(path) -> datasets.Dataset:
        path = get_data_path(path)
        
        dataset = []
        
        with open(path, 'r', encoding='utf-8') as f:
            file_data = json.load(f)
        
        for item in file_data:
            if "task" in item and "clues" in item and "answer" in item:
                clues_string = ""
                for i, clue in enumerate(item["clues"], 1):
                    clues_string += f"{i}. {clue}\n"
                clues_string = clues_string.rstrip("\n")  # Remove trailing newline
                    
                data = {
                    "task": item["task"],
                    "clues": clues_string,
                    "answer": item["answer"]
                }
                dataset.append(data)

        return DatasetDict({'test': Dataset.from_list(dataset), 'train': []})
    

@LOAD_DATASET.register_module()
class FunnyDatasetV2(BaseDataset):

    @staticmethod
    def load(path: str, num_repeats: int = 1) -> datasets.DatasetDict:
        """Load Funny dataset for pass@k mode.

        Note that you can use num_repeats > 1 when your model does not support
        `num_return_sequence` in generation. Otherwise, use the raw
        Funny dataset and set `num_return_sequence` in the model config to
        generate multiple responses for testing pass@k where k > 1.

        Args:
            path (str): Path to the dataset file.
            num_repeats (int): Number of times to repeat the test set.
                Defaults to 1. Must be >= 1.
        """
        if num_repeats < 1:
            raise ValueError("num_repeats must be at least 1.")

        path = get_data_path(path)
        
        processed_data_list = []
        
        with open(path, 'r', encoding='utf-8') as f:
            file_data = json.load(f)
        
        for item in file_data:
            if "task" in item and "clues" in item and "answer" in item:
                clues_string = ""
                for i, clue in enumerate(item["clues"], 1):
                    clues_string += f"{i}. {clue}\n"
                clues_string = clues_string.rstrip("\n")

                data = {
                    "task": item["task"],
                    "clues": clues_string,
                    "answer": item["answer"]
                }
                processed_data_list.append(data)

        # Create a single instance of the test dataset
        if not processed_data_list:
            test_dataset_single = Dataset.from_list([])
        else:
            test_dataset_single = Dataset.from_list(processed_data_list)

        # Repeat the test dataset num_repeats times
        if num_repeats > 0: # num_repeats is already asserted to be >= 1
            test_dataset_repeated = concatenate_datasets(
                [test_dataset_single] * num_repeats)
        else: # Should not happen due to assertion, but as a fallback
            test_dataset_repeated = test_dataset_single
            
        # FunnyDataset provides an empty train set
        train_dataset = Dataset.from_list([])

        return DatasetDict({'test': test_dataset_repeated, 'train': train_dataset})