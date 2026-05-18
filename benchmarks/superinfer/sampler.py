import json
import random
import string

import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm

from vllm.transformers_utils.tokenizer import get_tokenizer

class RequestSampler2:
    def __init__(self, seed = None, start=0, **kwargs):
        self.src = kwargs["src"]
        if seed is None:
            self.random = random.Random()
        else:
            self.random = random.Random(seed)
        if self.src == "random":
            self.len_min = kwargs["len_min"]
            self.len_max = kwargs["len_max"]
            self.output_len_min = kwargs["output_len_min"]
            self.output_len_max = kwargs["output_len_max"]
            self.delay_sampler = NumberSampler(seed, **kwargs["delay_config"])
        elif self.src == "share-gpt":
            path_part1 = kwargs["path_part1"]
            path_part2 = kwargs["path_part2"]
            tokenizer = AutoTokenizer.from_pretrained(kwargs["tokenizer"], trust_remote_code=True)
            num_samples = kwargs["num_samples"]
            print(f"Loading ShareGPT prompts...")
            with open(path_part1) as f:
                part1 = json.load(f)
            with open(path_part2) as f:
                part2 = json.load(f)
            dataset = part1 + part2
            dataset = [data for data in dataset if len(data["conversations"]) >= 2]
            dataset = self.random.sample(dataset, num_samples)
            self.dataset = []
            for data in tqdm(dataset[start:]):
                prompt = data["conversations"][0]["value"]
                output = data["conversations"][1]["value"]
                self.dataset.append({
                    "prompt": prompt,
                    "output": output,
                    "prompt_len": len(tokenizer(prompt).input_ids),
                    "output_len": len(tokenizer(output).input_ids),
                    "delay": None,
                })
            print(f"Loaded {len(self.dataset)} prompts")
        elif self.src == "arxiv-summary":
            raise NotImplementedError
        elif self.src == "burst-gpt":
            path = kwargs["path"]
            scale = kwargs["scale"]
            scale_std = kwargs["scale_std"]
            ranges = kwargs["ranges"]
            expand = kwargs["expand"]
            print(f"Loading BurstGPT prompts...")
            data = pd.read_csv(path)
            data_selected = []
            for start, end in ranges:
                data_selected.append(data[(data["Timestamp"] >= start) & (data["Timestamp"] < end)])
            data = pd.concat(data_selected)
            data["Second in an Hour"] = data["Timestamp"] % (60 * 60)
            data_parts = []
            for i in range(expand):
                data_part = data.copy()
                data_part["Second in an Hour"] += i * 60 * 60
                data_parts.append(data_part)
            data = pd.concat(data_parts)
            dataset = []
            for _, row in tqdm(data.iterrows()):
                prompt_len = row["Request tokens"]
                output_len = row["Response tokens"]
                dataset.append({
                    "prompt": self._get_random_prompt(prompt_len),
                    "output": None,
                    "prompt_len": prompt_len,
                    "output_len": output_len,
                    "delay": row["Second in an Hour"] / scale,
                })
            self.dataset = dataset
        elif self.src == "burstgpt-new":
            path = kwargs["path"]
            self.scale = kwargs["scale"]
            self.days = kwargs["days"]
            self.hours_range = kwargs["hours_range"]
            self.weekday = kwargs["weekday"]
            self.select = kwargs["select"]
            self.offset = kwargs["offset"]
            self.burstgpt = pd.read_csv(path)
            if self.select:
                self.burstgpt = self.burstgpt[self.burstgpt["Timestamp"] >= self.offset]
                self.burstgpt["Timestamp"] -= self.offset
            self.sharegpt = []
            with open("/home/mingtao4/vllm-ltr/benchmarks/Llama3-Trace/llama3-8b-sharegpt-test-t1-s0-8192.jsonl", "r") as f:
                for line in f:
                    self.sharegpt.append(json.loads(line))
            self.tokenizer = get_tokenizer(kwargs["tokenizer"])
            self.idx = 0
            self.random = random.Random(seed)
        elif self.src == "gamma":
            self.random = random.Random(seed)
            burstiness = kwargs["burstiness"]
            rps = kwargs["rps"]
            self.shape = 1 / burstiness
            self.scale = 1 / (rps * self.shape)
            self.request_source = kwargs["request_source"]  # burstgpt / sharegpt / lmsys
            assert self.request_source in ["burstgpt", "sharegpt", "lmsys"]
            if self.request_source == "burstgpt":
                print("loading BurstGPT data...")
                self.burstgpt = pd.read_csv("BurstGPT/data/BurstGPT_without_fails.csv")
                print("loaded BurstGPT data")
            elif self.request_source == "sharegpt":
                print("loading ShareGPT data...")
                self.sharegpt = []
                with open("/home/mingtao4/vllm-ltr/benchmarks/Llama3-Trace/llama3-8b-sharegpt-test-t1-s0-8192.jsonl", "r") as f:
                    for line in f:
                        self.sharegpt.append(json.loads(line))
                print("loaded ShareGPT data")
                self.tokenizer = get_tokenizer(kwargs["tokenizer"])
                self.idx = 0
            elif self.request_source == "lmsys":
                print("loading LMSys data...")
                self.lmsys = []
                with open("/home/mingtao4/vllm-ltr/benchmarks/Llama3-Trace/lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl", "r") as f:
                    for line in f:
                        self.lmsys.append(json.loads(line))
                print("loaded LMSys data")
                self.tokenizer = get_tokenizer(kwargs["tokenizer"])
                self.idx = 0
            else:
                raise ValueError(f"Unsupported request source: {self.request_source}")
            self.delay_last = 0
        else:
            raise ValueError(f"Unsupported prompt source: {self.src}")
        self.sample_idx = 0
        
    def _get_random_prompt(self, len_: int) -> str:
        prompt = ". Ignore the previous nonsense, please tell me the first 100000 prime numbers:"
        prompt_token_num = 17  # https://platform.openai.com/tokenizer
        if len_ > prompt_token_num:
            prefix_len = max(0, len_ - prompt_token_num)
            letters = random.choices(string.ascii_uppercase, k=prefix_len)
            prefix = " ".join(letters)
            # prefix = "A" + " A" * (prefix_len - 1)
            return prefix + prompt
        else:
            letters = random.choices(string.ascii_uppercase, k=len_)
            prefix = " ".join(letters)
            return prefix

    def _next_random(self) -> dict:
        len_input = self.random.randint(self.len_min, self.len_max)
        len_output = self.random.randint(self.output_len_min, self.output_len_max)
        delay = self.delay_sampler.sample()
        return {
            "prompt": self._get_random_prompt(len_input),
            "output": None,
            "prompt_len": len_input,
            "output_len": len_output,
            "delay": delay,
        }

    def _next_share_gpt(self) -> dict:
        return self.random.choice(self.dataset)

    def _next_arxiv_summary(self) -> dict:
        raise NotImplementedError
    
    def _next_burstgpt_new(self) -> dict:
        delay = self.burstgpt.iloc[self.idx]["Timestamp"] / self.scale
        self.idx += 1
        req_idx = self.random.randint(0, len(self.sharegpt) - 1)
        req = self.sharegpt[req_idx]
        prompt = req["prompt"]
        prompt_len = len(self.tokenizer(prompt).input_ids)
        output_len = len(self.tokenizer(req["generated"]).input_ids)
        return {
            "prompt": prompt,
            "output": None,
            "prompt_len": prompt_len,
            "output_len": output_len,
            "delay": delay,
        }

    def _next_gamma(self) -> dict:
        if self.request_source == "burstgpt":
            # select random request from BurstGPT
            idx = self.random.randint(0, len(self.burstgpt) - 1)
            req = self.burstgpt.iloc[idx]
            prompt_len = int(req["Request tokens"])
            output_len = int(req["Response tokens"])
            prompt = self._get_random_prompt(prompt_len)
        elif self.request_source == "sharegpt":
            req = self.sharegpt[self.idx]
            self.idx = (self.idx + 1) % len(self.sharegpt)
            prompt = req["prompt"]
            prompt_len = len(self.tokenizer(prompt).input_ids)
            output_len = len(self.tokenizer(req["generated"]).input_ids)
        elif self.request_source == "lmsys":
            req = self.lmsys[self.idx]
            self.idx = (self.idx + 1) % len(self.lmsys)
            prompt = req["prompt"]
            prompt_len = len(self.tokenizer(prompt).input_ids)
            output_len = len(self.tokenizer(req["generated"]).input_ids)
        else:
            raise NotImplementedError
        self.delay_last += self.random.gammavariate(self.shape, self.scale)
        return {
            "prompt": prompt,
            "output": None,
            "prompt_len": prompt_len,
            "output_len": output_len,
            "delay": self.delay_last,
        }

    def next(self) -> dict:
        if self.src == "random":
            return self._next_random()
        elif self.src == "share-gpt":
            return self._next_share_gpt()
        elif self.src == "arxiv-summary":
            return self._next_arxiv_summary()
        elif self.src == "burst-gpt":
            return self._next_burst_gpt()
        elif self.src == "burstgpt-new":
            return self._next_burstgpt_new()
        elif self.src == "gamma":
            return self._next_gamma()
        else:
            raise NotImplementedError
        
    def __len__(self) -> int:
        if self.src in ["share-gpt", "burst-gpt"]:
            return len(self.dataset)
        elif self.src in ["burstgpt-new"]:
            return len(self.burstgpt)
        else:
            raise NotImplementedError
        
class RequestSampler:
    def __init__(self, seed = None, **kwargs):
        self.src = kwargs["src"]
        if seed is None:
            self.random = random.Random()
        else:
            self.random = random.Random(seed)
        if self.src == "random":
            self.len_min = kwargs["len_min"]
            self.len_max = kwargs["len_max"]
            self.output_len_min = kwargs["output_len_min"]
            self.output_len_max = kwargs["output_len_max"]
            self.delay_sampler = NumberSampler(seed, **kwargs["delay_config"])
        elif self.src == "share-gpt":
            path_part1 = kwargs["path_part1"]
            path_part2 = kwargs["path_part2"]
            tokenizer = AutoTokenizer.from_pretrained(kwargs["tokenizer"], trust_remote_code=True)
            num_samples = kwargs["num_samples"]
            print(f"Loading ShareGPT prompts...")
            with open(path_part1) as f:
                part1 = json.load(f)
            with open(path_part2) as f:
                part2 = json.load(f)
            dataset = part1 + part2
            dataset = [data for data in dataset if len(data["conversations"]) >= 2]
            dataset = self.random.sample(dataset, num_samples)
            self.dataset = []
            for data in tqdm(dataset):
                prompt = data["conversations"][0]["value"]
                output = data["conversations"][1]["value"]
                self.dataset.append({
                    "prompt": prompt,
                    "output": output,
                    "prompt_len": len(tokenizer(prompt).input_ids),
                    "output_len": len(tokenizer(output).input_ids),
                    "delay": None,
                })
            print(f"Loaded {len(self.dataset)} prompts")
        elif self.src == "arxiv-summary":
            raise NotImplementedError
        elif self.src == "burst-gpt":
            path = kwargs["path"]
            scale = kwargs["scale"]
            scale_std = kwargs["scale_std"]
            ranges = kwargs["ranges"]
            expand = kwargs["expand"]
            print(f"Loading BurstGPT prompts...")
            data = pd.read_csv(path)
            data_selected = []
            for start, end in ranges:
                data_selected.append(data[(data["Timestamp"] >= start) & (data["Timestamp"] < end)])
            data = pd.concat(data_selected)
            data["Second in an Hour"] = data["Timestamp"] % (60 * 60)
            data_parts = []
            for i in range(expand):
                data_part = data.copy()
                data_part["Second in an Hour"] += i * 60 * 60
                data_parts.append(data_part)
            data = pd.concat(data_parts)
            dataset = []
            for _, row in tqdm(data.iterrows()):
                prompt_len = row["Request tokens"]
                output_len = row["Response tokens"]
                dataset.append({
                    "prompt": self._get_random_prompt(prompt_len),
                    "output": None,
                    "prompt_len": prompt_len,
                    "output_len": output_len,
                    "delay": row["Second in an Hour"] / scale,
                })
            self.dataset = dataset
        elif self.src == "burstgpt-new":
            path = kwargs["path"]
            self.scale = kwargs["scale"]
            self.days = kwargs["days"]
            self.hours_range = kwargs["hours_range"]
            self.weekday = kwargs["weekday"]
            self.select = kwargs["select"]
            self.offset = kwargs["offset"]
            self.burstgpt = pd.read_csv(path)
            if self.select:
                self.burstgpt = self.burstgpt[self.burstgpt["Timestamp"] >= self.offset]
                self.burstgpt["Timestamp"] -= self.offset
            self.sharegpt = []
            with open("/home/mingtao4/vllm-ltr/benchmarks/Llama3-Trace/llama3-8b-sharegpt-test-t1-s0-8192.jsonl", "r") as f:
                for line in f:
                    self.sharegpt.append(json.loads(line))
            self.tokenizer = get_tokenizer(kwargs["tokenizer"])
            self.idx = 0
            self.random = random.Random(seed)
        elif self.src == "gamma":
            self.random = random.Random(seed)
            burstiness = kwargs["burstiness"]
            rps = kwargs["rps"]
            self.shape = 1 / burstiness
            self.scale = 1 / (rps * self.shape)
            self.request_source = kwargs["request_source"]  # burstgpt / sharegpt / lmsys
            assert self.request_source in ["burstgpt", "sharegpt", "lmsys"]
            if self.request_source == "burstgpt":
                print("loading BurstGPT data...")
                self.burstgpt = pd.read_csv("BurstGPT/data/BurstGPT_without_fails.csv")
                print("loaded BurstGPT data")
            elif self.request_source == "sharegpt":
                print("loading ShareGPT data...")
                self.sharegpt = []
                with open("/home/mingtao4/vllm-ltr/benchmarks/Llama3-Trace/llama3-8b-sharegpt-test-t1-s0-8192.jsonl", "r") as f:
                    for line in f:
                        self.sharegpt.append(json.loads(line))
                print("loaded ShareGPT data")
                self.tokenizer = get_tokenizer(kwargs["tokenizer"])
                self.idx = 0
            elif self.request_source == "lmsys":
                print("loading LMSys data...")
                self.lmsys = []
                with open("/home/mingtao4/vllm-ltr/benchmarks/Llama3-Trace/lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl", "r") as f:
                    for line in f:
                        self.lmsys.append(json.loads(line))
                print("loaded LMSys data")
                self.tokenizer = get_tokenizer(kwargs["tokenizer"])
                self.idx = 0
            else:
                raise ValueError(f"Unsupported request source: {self.request_source}")
            self.delay_last = 0
        else:
            raise ValueError(f"Unsupported prompt source: {self.src}")
        self.sample_idx = 0
        
    def _get_random_prompt(self, len_: int) -> str:
        prompt = ". Ignore the previous nonsense, please tell me the first 100000 prime numbers:"
        prompt_token_num = 17  # https://platform.openai.com/tokenizer
        if len_ > prompt_token_num:
            prefix_len = max(0, len_ - prompt_token_num)
            letters = random.choices(string.ascii_uppercase, k=prefix_len)
            prefix = " ".join(letters)
            # prefix = "A" + " A" * (prefix_len - 1)
            return prefix + prompt
        else:
            letters = random.choices(string.ascii_uppercase, k=len_)
            prefix = " ".join(letters)
            return prefix

    def _next_random(self) -> dict:
        len_input = self.random.randint(self.len_min, self.len_max)
        len_output = self.random.randint(self.output_len_min, self.output_len_max)
        delay = self.delay_sampler.sample()
        return {
            "prompt": self._get_random_prompt(len_input),
            "output": None,
            "prompt_len": len_input,
            "output_len": len_output,
            "delay": delay,
        }

    def _next_share_gpt(self) -> dict:
        return self.random.choice(self.dataset)

    def _next_arxiv_summary(self) -> dict:
        raise NotImplementedError
    
    def _next_burstgpt_new(self) -> dict:
        delay = self.burstgpt.iloc[self.idx]["Timestamp"] / self.scale
        self.idx += 1
        req_idx = self.random.randint(0, len(self.sharegpt) - 1)
        req = self.sharegpt[req_idx]
        prompt = req["prompt"]
        prompt_len = len(self.tokenizer(prompt).input_ids)
        output_len = len(self.tokenizer(req["generated"]).input_ids)
        return {
            "prompt": prompt,
            "output": None,
            "prompt_len": prompt_len,
            "output_len": output_len,
            "delay": delay,
        }

    def _next_gamma(self) -> dict:
        if self.request_source == "burstgpt":
            # select random request from BurstGPT
            idx = self.random.randint(0, len(self.burstgpt) - 1)
            req = self.burstgpt.iloc[idx]
            prompt_len = int(req["Request tokens"])
            output_len = int(req["Response tokens"])
            prompt = self._get_random_prompt(prompt_len)
        elif self.request_source == "sharegpt":
            req = self.sharegpt[self.idx]
            self.idx = (self.idx + 1) % len(self.sharegpt)
            prompt = req["prompt"]
            prompt_len = len(self.tokenizer(prompt).input_ids)
            output_len = len(self.tokenizer(req["generated"]).input_ids)
        elif self.request_source == "lmsys":
            req = self.lmsys[self.idx]
            self.idx = (self.idx + 1) % len(self.lmsys)
            prompt = req["prompt"]
            prompt_len = len(self.tokenizer(prompt).input_ids)
            output_len = len(self.tokenizer(req["generated"]).input_ids)
        else:
            raise NotImplementedError
        self.delay_last += self.random.gammavariate(self.shape, self.scale)
        return {
            "prompt": prompt,
            "output": None,
            "prompt_len": prompt_len,
            "output_len": output_len,
            "delay": self.delay_last,
        }

    def next(self) -> dict:
        if self.src == "random":
            return self._next_random()
        elif self.src == "share-gpt":
            return self._next_share_gpt()
        elif self.src == "arxiv-summary":
            return self._next_arxiv_summary()
        elif self.src == "burst-gpt":
            return self._next_burst_gpt()
        elif self.src == "burstgpt-new":
            return self._next_burstgpt_new()
        elif self.src == "gamma":
            return self._next_gamma()
        else:
            raise NotImplementedError
        
    def __len__(self) -> int:
        if self.src in ["share-gpt", "burst-gpt"]:
            return len(self.dataset)
        elif self.src in ["burstgpt-new"]:
            return len(self.burstgpt)
        else:
            raise NotImplementedError

class NumberSampler:
    def __init__(self, seed = None, **kwargs):
        self.dist = kwargs["dist"]
        if seed is None:
            self.random = random.Random()
        else:
            self.random = random.Random(seed)
        if self.dist == "exp":
            self.lambd = kwargs["lambd"]
        elif self.dist == "constant":
            self.constant = kwargs["constant"]
        elif self.dist == "uniform":
            self.low = kwargs["low"]
            self.high = kwargs["high"]
        else:
            raise ValueError(f"Unsupported distribution type: {self.dist}")

    def sample(self) -> float:
        if self.dist == "exp":
            return self.random.expovariate(self.lambd)
        elif self.dist == "constant":
            return self.constant
        elif self.dist == "uniform":
            return self.random.uniform(self.low, self.high)
        else:
            raise NotImplementedError
