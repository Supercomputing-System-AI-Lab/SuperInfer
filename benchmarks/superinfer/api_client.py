import argparse
import asyncio
import glob
import json
import os
import sys
import time
from typing import Optional

import aiohttp
import numpy as np
import orjson

from sampler import RequestSampler

class RequestTracer:
    
    def __init__(self, url, payload, id):
        self.url = url
        self.payload = payload
        self.header = {"User-Agent": "Requests Tracer"}
        self.id = id

    async def run(self, delay: Optional[int] = None):
        if delay is not None:
            await asyncio.sleep(delay)
        record = {
            "prompt": self.payload["prompt"],
            "max_tokens": self.payload["max_tokens"],
            "observability": [],
            "submission_time": time.time(),
            "times": [],
            "real_token_num": None,
            "worker_idx": None,
            "text_output": '',
        }
        timeout = aiohttp.ClientTimeout(total=86400)
        # print(f"request {self.id} sent")
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url=self.url, json=self.payload) as resp:
                buffer = b""
                async for chunk in resp.content.iter_any():
                    buffer += chunk
                    start = 0
                    while len(buffer) > 0:
                        idx = buffer.find(b"\0", start)
                        if idx == -1:
                            start = len(buffer)
                            break
                        line, buffer, start = buffer[:idx], buffer[idx+1:], 0
                        # token_info = json.loads(line)
                        token_info = orjson.loads(line)
                        if "observability" in token_info:
                            record["observability"].append(token_info["observability"])
                        if "time" in token_info:
                            record["times"].append(token_info["time"])

                        if record["worker_idx"] is None and "worker_idx" in token_info:
                            record["worker_idx"] = token_info["worker_idx"]

                record["text_output"] = token_info["text"]
                record["real_token_num"] = len(record["observability"])
        # print(f"request {self.id} finished")
        return record


class RequestSubmitter:

    def __init__(self, server_url, num_req, total_time, request_sampler_config, delay_sampler_config, seed):
        self.base_url = server_url
        self.url = server_url + "/generate"
        self.request_sampler = RequestSampler(seed, **request_sampler_config)
        self.num_req = num_req
        self.total_time = total_time
        self._finished = False
        self.records = []
        self.stats = []

    async def stats_coro(self):
        interval = 1
        client = aiohttp.ClientSession()
        print("fetching stats ...")
        while not self._finished:
            await asyncio.sleep(interval)
            resp = await client.get(self.base_url + "/stats")
            stats_dict = await resp.json()
            stats_dict["time"] = time.time()
            self.stats.append(stats_dict)

    async def run(self):
        
        if self.num_req is None:
            self.num_req = len(self.request_sampler)
        tasks = []
        tracers = []
        for i in range(self.num_req):
            request_info = self.request_sampler.next()
            delay = request_info["delay"]
            if self.total_time is not None and delay > self.total_time:
                break
            max_tokens = request_info["output_len"]
            if max_tokens is None:
                max_tokens = 16384
            payload = {
                "prompt": request_info["prompt"],
                "n": 1,
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "ignore_eos": True,
                "stream": True,
                # "stream": False,
                "debug_id": i,
            }
            tracer = RequestTracer(url=self.url, payload=payload, id=i)
            tracers.append((tracer, delay))

        print("sending requests ...")
        stats_task = asyncio.create_task(self.stats_coro())
        for tracer, delay in tracers:
            task = asyncio.create_task(tracer.run(delay=delay))
            tasks.append(task)

        print("all requests sent, waiting for finished ...")
        self.records = await asyncio.gather(*tasks)
        self._finished = True
        await stats_task
        print("all requests finished, results collected")

def draw_stats_v2(dir_to_analyze: str):
    record_file_path = dir_to_analyze + "/records-tmp.json"
    stats_file_path = dir_to_analyze + "/stats-tmp.json"
    result = []
    with open(record_file_path, 'r') as f:
        records = json.load(f)
    with open(stats_file_path, 'r') as f:
        stats = json.load(f)
    ttfts = [r["times"][0] - r["submission_time"] for r in records]
    tbts = []
    for r in records:
        tbts.extend([t - t_prev for t, t_prev  in zip(r["times"][1:], r["times"][:-1])])
    result.append("Average TTFT: " + str(np.mean(ttfts)))
    result.append("TTFT P50: " + str(np.percentile(ttfts, 50)))
    result.append("TTFT P90: " + str(np.percentile(ttfts, 90)))
    result.append("TTFT P99: " + str(np.percentile(ttfts, 99)))

    result.append("Average TBT: " + str(np.mean(tbts)))
    result.append("TBT P50: " + str(np.percentile(tbts, 50)))
    result.append("TBT P90: " + str(np.percentile(tbts, 90)))
    result.append("TBT P99: " + str(np.percentile(tbts, 99)))

    return result

def AIO(dir_to_analyze):
    result = draw_stats_v2(dir_to_analyze)

    result_text_path = dir_to_analyze + "/result.txt"
    with open(result_text_path, 'w') as f:
        for line in result:
            f.write(line + "\n")
    
def main(args):
    if args.output_dir[-1] == "/":
        output_dir = args.output_dir[:-1]
    else:
        output_dir = args.output_dir
    num_req = args.num_req
    
    total_time = args.total_time
    tokenizer = args.tokenizer
    request_sampler_config = {
        "src": "gamma",
        "burstiness": args.burstiness,
        "rps": args.rps,
        "request_source": args.dataset,
        "tokenizer": tokenizer,
    }
    dir_prefix = output_dir + "/data_numreq=" + str(num_req) + "_rps=" + str(request_sampler_config['rps']) + "_burstiness=" + str(request_sampler_config['burstiness']) + "_tokenizer=" + tokenizer.split('/')[1] + "_" + "_dataset=" + args.dataset
    
    if os.path.exists(output_dir):
        for path in glob.glob(f'{output_dir}/*'):
            if dir_prefix in path:
                print("Everything Finished.")
                return
    seed = 42
    port = args.port
    server_url = "http://localhost:" + port

    submitter = RequestSubmitter(
        server_url,
        num_req,
        total_time,
        request_sampler_config,
        None,
        seed,
    )

    asyncio.run(submitter.run())

    
    data_dir = output_dir + "/data_numreq=" + str(num_req) + "_rps=" + str(request_sampler_config['rps']) + "_burstiness=" + str(request_sampler_config['burstiness']) + "_tokenizer=" + tokenizer.split('/')[1] + "_" + "_dataset=" + args.dataset + "_" + time.strftime("%Y%m%d-%H%M%S")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    with open(data_dir + "/records-tmp.json", "w") as f:
        json.dump(submitter.records, f, indent=4)
    with open(data_dir + "/stats-tmp.json", "w") as f:
        json.dump(submitter.stats, f, indent=4)

    AIO(data_dir)
    print("Everything Finished.")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_req",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--total_time",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--burstiness",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--rps",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
    )
    
    parser.add_argument(
        "--port",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )
    args = parser.parse_args(sys.argv[1:])
    main(args)
