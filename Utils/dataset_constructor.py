import torch
import datasets
import itertools
from torch.utils.data import IterableDataset, get_worker_info
import json
import os.path as osp
from typing import Union
import warnings
from datasets.distributed import split_dataset_by_node
    

def dataloader_constructor(
        dataset,
        batch_size,
        max_length,
        tokenizer, 
        num_workers,
        train_on_prompts=True,
        add_eos_token=True,
        local_rank=0,
        world_size=1 
    ):
    shulffe_seed = 42
    if dataset.lower() == 'commonsense':
        data = datasets.load_dataset("/path/to/dataset", split="train") #commonsense170k
        data: datasets.Dataset = data.shuffle(seed=shulffe_seed)
        data = split_dataset_by_node(
            data, rank=local_rank, world_size=world_size,
        )
        dataset = PreprocessedCommonSenceDataset(
            data, 
            tokenizer, 
            batch_size, 
            max_length,
            train_on_prompts=train_on_prompts, 
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=num_workers)
    elif dataset.lower() == 'mmlu':
        mmlu_train = datasets.load_dataset("/path/to/dataset", "abstract_algebra", split="auxiliary_train")
        mmlu_train: datasets.Dataset = mmlu_train.shuffle(seed=shulffe_seed)
        mmlu_train = split_dataset_by_node(
            mmlu_train, rank=local_rank, world_size=world_size,
        )
        dataset = PreprocessedMMLUDataset(
            mmlu_train, 
            tokenizer, 
            batch_size, 
            max_length,
            train_on_prompts=train_on_prompts, 
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=num_workers)
    elif dataset.lower() == 'gsm8k':
        gsm8k_train = datasets.load_dataset("/path/to/dataset", "main", split="train")
        gsm8k_train: datasets.Dataset = gsm8k_train.shuffle(seed=shulffe_seed)
        gsm8k_train = split_dataset_by_node(
            gsm8k_train, rank=local_rank, world_size=world_size,
        )
        dataset = PreprocessedGSM8KDataset(
            gsm8k_train, 
            tokenizer, 
            batch_size, 
            max_length
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=num_workers)
    return dataset, dataloader

class PreprocessedIterableDataset(IterableDataset):
    def __init__(self, data, tokenizer, batch_size, max_length):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

    def _format_batch(self, batch):
        data = {}
        for key in batch[0].keys():
            data[key] = torch.stack([item[key].squeeze(0) for item in batch])
        return data
    
    def _get_iter_date(self):
        worker_info = get_worker_info()
        seed = torch.randint(0, 1024, (1,)).item()
        if worker_info is None:
            iter_data = iter(self.data.shuffle(seed=seed))
        else:
            # If using DataLoader workers, yield a subset of the data for this worker
            worker_id = worker_info.id
            # print(worker_id)
            num_workers = worker_info.num_workers
            iter_data = itertools.islice(self.data.shuffle(seed=seed), worker_id, None, num_workers)
        return iter_data
    
    def _tokenize(self, prompt, padding="max_length", add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding=padding,
            return_tensors="pt",
        )
        result["input_ids"] = result["input_ids"].squeeze()
        return result


class PreprocessedCommonSenceDataset(PreprocessedIterableDataset):
    def __init__(self, data, tokenizer, batch_size, max_length, train_on_prompts=True):
        super().__init__(data, tokenizer, batch_size, max_length)

        self.NO_QA_TEMPLATE_WITH_INPUT = """
        ### Instruction:
        {instruction}

        ### Input:
        {input}

        ### Response:
        """

        self.NO_QA_TEMPLATE_WITHOUT_INPUT = """
        ### Instruction:
        {instruction}

        ### Response:
        """ # noqa: E501
        self.train_on_prompts = train_on_prompts

    def __iter__(self):
        batch = []
        while True:
            iter_data = self._get_iter_date()
            # print(iter_data)
            for idx, example in enumerate(iter_data):
                tokenized_example = self.generate_and_tokenize_prompt(example)
                batch.append(tokenized_example)
                if len(batch) == self.batch_size:
                    yield self._format_batch(batch)
                    batch = []

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        if input:
            res = self.NO_QA_TEMPLATE_WITH_INPUT.format(
                instruction=instruction, input=input
            )
        else:
            res = self.NO_QA_TEMPLATE_WITHOUT_INPUT.format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        # if self._verbose:
        #     print(res)
        return res

    def generate_and_tokenize_prompt(self, data_point):
        full_prompt = self.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = self.tokenizer(
            full_prompt,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        # add eos token if not present
        if (
            tokenized_full_prompt["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(tokenized_full_prompt["input_ids"]) < self.max_length
        ):
            tokenized_full_prompt["input_ids"].append(self.tokenizer.eos_token_id)
            tokenized_full_prompt["attention_mask"].append(1)
             
        tokenized_full_prompt = self.tokenizer.pad(
            tokenized_full_prompt, 
            max_length=self.max_length, 
            padding="max_length", 
            return_tensors='pt'
        )
        tokenized_full_prompt["input_ids"] = tokenized_full_prompt["input_ids"].squeeze()
        tokenized_full_prompt["labels"] = tokenized_full_prompt["input_ids"].clone()
        if not self.train_on_prompts:
            user_prompt = self.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = self.tokenizer(
                user_prompt,
                truncation=True,
                padding=False,
                return_tensors=None,
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            if user_prompt_len >= self.max_length:
                warnings.warn("User prompt is too long")
            ignored_tokens = torch.tensor([-100] * user_prompt_len, dtype=tokenized_full_prompt["labels"].dtype)
            tokenized_full_prompt["labels"] = torch.concat(
                [ignored_tokens, tokenized_full_prompt["input_ids"][user_prompt_len:]], dim=0
            ) # could be sped up, probably
        tokenized_full_prompt["labels"][tokenized_full_prompt["labels"] == self.tokenizer.pad_token_id] = -100
        return tokenized_full_prompt
    
class PreprocessedMMLUDataset(PreprocessedIterableDataset):
    def __init__(
            self, 
            data, 
            tokenizer, 
            batch_size, 
            max_length,
            train_on_prompts: bool = True,
            add_eos_token: bool = True
        ):
        super().__init__(data, tokenizer, batch_size, max_length)
        self.choices = ["A", "B", "C", "D"]
        self.train_on_prompts = train_on_prompts

    def __iter__(self):
        batch = []
        while True:
            iter_data = self._get_iter_date()
            for idx, example in enumerate(iter_data):
                tokenized_example = self.generate_and_tokenize_prompt(example)
                batch.append(tokenized_example)
                if len(batch) == self.batch_size:
                    yield self._format_batch(batch)
                    batch = []

    def generate_and_tokenize_prompt(self, data_point):
        prompt = "The following are multiple choice questions (with answers).\n\n"
        prompt += "Question: {}\n".format(data_point['question'])
        k = len(data_point['choices'])
        for j in range(k):
            prompt += "\n{}. {}".format(self.choices[j], data_point['choices'][j])
        prompt += "\nAnswer:"
        full_prompt = prompt + "{}".format(self.choices[data_point['answer']])

        tokenized_full_prompt = self.tokenizer(
            full_prompt,
            max_length=self.max_length, 
            truncation=True,
            padding="max_length",
            return_tensors='pt',
        )
        tokenized_full_prompt["input_ids"] = tokenized_full_prompt["input_ids"].squeeze()
        tokenized_full_prompt["labels"] = tokenized_full_prompt["input_ids"].clone()
        if not self.train_on_prompts:
            tokenized_question_prompt = self.tokenizer(
                prompt,
                truncation=True,
                padding=False,
                return_tensors=None,
            )
            user_prompt_len = len(tokenized_question_prompt["input_ids"])
            if user_prompt_len >= self.max_length:
                # warnings.warn("User prompt is too long")
                print("User prompt is too long")
            ignored_tokens = torch.tensor([-100] * user_prompt_len, dtype=tokenized_full_prompt["labels"].dtype)
            tokenized_full_prompt["labels"] = torch.concat(
                [ignored_tokens, tokenized_full_prompt["input_ids"][user_prompt_len:]], dim=0
            ) # could be sped up, probably    
        tokenized_full_prompt["labels"][tokenized_full_prompt["labels"] == self.tokenizer.pad_token_id] = -100
        assert len(tokenized_full_prompt["input_ids"]) == self.max_length, f"Input_ids length is {len(tokenized_full_prompt['input_ids'])}"
        return tokenized_full_prompt
    

class PreprocessedGSM8KDataset(PreprocessedIterableDataset):
    def __init__(self, data, tokenizer, batch_size, max_length):
        super().__init__(data, tokenizer, batch_size, max_length)

    def __iter__(self):
        batch = []
        while True:
            iter_data = self._get_iter_date()
            for idx, example in enumerate(iter_data):
                full_prompt = self._generate_prompt(example)
                tokenized_full_prompt = self.tokenizer(
                    full_prompt,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                # tokenized_example = self._tokenize(prompt=example["text"])
                tokenized_full_prompt["input_ids"] = tokenized_full_prompt["input_ids"].squeeze()
                tokenized_full_prompt["labels"] = tokenized_full_prompt["input_ids"].clone()
                tokenized_full_prompt["labels"][tokenized_full_prompt["labels"] == self.tokenizer.pad_token_id] = -100
                batch.append(tokenized_full_prompt)
                if len(batch) == self.batch_size:
                    yield self._format_batch(batch)
                    batch = []

    def _generate_prompt(self, data_point):
        return "Question: {question}\nAnswer: {answer}".format(
            question=data_point["question"], answer=data_point["answer"]
        )