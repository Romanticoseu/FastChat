"""
A model worker that executes the model based on BigDL-LLM.

See documentations at docs/bigdlllm_integration.md
"""

import argparse
import asyncio
import json
import os
from typing import List, Optional
from bigdl.llm.utils.common import invalidInputError
from bigdl.llm.utils import load_model

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse

from fastchat.model.model_adapter import (
    add_model_args,
    get_generate_stream_function,
)
from fastchat.modules.gptq import GptqConfig
from fastchat.modules.awq import AWQConfig
from fastchat.serve.base_model_worker import BaseModelWorker
from fastchat.constants import WORKER_HEART_BEAT_INTERVAL, ErrorCode, SERVER_ERROR_MSG
from fastchat.utils import get_context_length
from fastchat.serve.model_worker import (
    logger,
    worker_id,
)

import torch
import uvicorn

app = FastAPI()
class BigDLLLMWorker(BaseModelWorker):
    def __init__(
        self, 
        controller_addr: str, 
        worker_addr: str, 
        worker_id: str, 
        model_path: str, 
        model_names: List[str], 
        limit_worker_concurrency: int, 
        no_register: bool,
        device: str,
        num_gpus: int,
        max_gpu_memory: str,
        low_bit: str,
        low_cpu_mem_usage: bool = False,
        cpu_offloading: bool = False,
        gptq_config: Optional[GptqConfig] = None,
        awq_config: Optional[AWQConfig] = None,
        stream_interval: int = 2,
        conv_template: str = None,
    ):
        super().__init__(
            controller_addr,
            worker_addr, 
            worker_id, 
            model_path, 
            model_names, 
            limit_worker_concurrency, 
            conv_template, 
        )
        
        logger.info(
            f"Loading the model {self.model_names} on worker {worker_id}, worker type: BigDLLLM worker..."
        )
        from bigdl.llm.utils import load_model
        self.model, self.tokenizer = load_model(
            model_path,
            device=device,
            num_gpus=num_gpus,
            max_gpu_memory=max_gpu_memory,
            low_bit=low_bit,
            cpu_offloading=cpu_offloading,
            gptq_config=gptq_config,
            awq_config=awq_config,
            debug=True
        )
        self.device = device
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.context_len = get_context_length(self.model.config)
        self.generate_stream_func = get_generate_stream_function(self.model, model_path)
        self.stream_interval = stream_interval
        
        if not no_register:
            self.init_heart_beat()

    def generate_stream_gate(self, params):
        self.call_ct += 1

        try:
            for output in self.generate_stream_func(
                self.model,
                self.tokenizer,
                params,
                self.device,
                self.context_len,
                self.stream_interval,
            ):
                ret = {
                    "text": output["text"],
                    "error_code": 0,
                }
                if "usage" in output:
                    ret["usage"] = output["usage"]
                if "finish_reason" in output:
                    ret["finish_reason"] = output["finish_reason"]
                if "logprobs" in output:
                    ret["logprobs"] = output["logprobs"]
                yield json.dumps(ret).encode() + b"\0"
        except torch.cuda.OutOfMemoryError as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
            yield json.dumps(ret).encode() + b"\0"
        except (ValueError, RuntimeError) as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
            yield json.dumps(ret).encode() + b"\0"
        
    def generate_gate(self, params):
        for x in self.generate_stream_gate(params):
            pass
        return json.loads(x[:-1].decode())
        
    @torch.inference_mode()
    def get_embeddings(self, params):
        self.call_ct += 1

        try:
            tokenizer = self.tokenizer
            is_llama = "llama" in str(
                type(self.model)
            )  # llama supports batch inference
            is_chatglm = "chatglm" in str(type(self.model))
            is_t5 = "t5" in str(type(self.model))
            is_bert = "bert" in str(type(self.model))

            if is_llama:
                encoding = tokenizer.batch_encode_plus(
                    params["input"], padding=True, return_tensors="pt"
                )
                input_ids = encoding["input_ids"].to(self.device)
                attention_mask = encoding["attention_mask"].to(self.device)
                model_output = self.model(
                    input_ids, attention_mask, output_hidden_states=True
                )
                data = model_output.hidden_states[-1]
                mask = attention_mask.unsqueeze(-1).expand(data.size()).float()
                masked_embeddings = data * mask
                sum_embeddings = torch.sum(masked_embeddings, dim=1)
                seq_length = torch.sum(mask, dim=1)
                embedding = sum_embeddings / seq_length
                normalized_embeddings = F.normalize(embedding, p=2, dim=1)
                ret = {
                    "embedding": normalized_embeddings.tolist(),
                    "token_num": torch.sum(attention_mask).item(),
                }
            elif is_bert:
                embedding = []
                token_num = 0
                for text in params["input"]:
                    input_ids = tokenizer.encode(text, return_tensors="pt").to(
                        self.device
                    )
                    model_output = self.model(input_ids)
                    data = model_output[0][:, 0]
                    data = F.normalize(torch.mean(data, dim=0), p=2, dim=0)
                    embedding.append(data.tolist())
                    token_num += len(input_ids[0])
                ret = {
                    "embedding": embedding,
                    "token_num": token_num,
                }
            else:
                embedding = []
                token_num = 0
                for text in params["input"]:
                    input_ids = tokenizer.encode(text, return_tensors="pt").to(
                        self.device
                    )
                    if is_t5:
                        model_output = self.model(
                            input_ids, decoder_input_ids=input_ids
                        )
                    else:
                        model_output = self.model(input_ids, output_hidden_states=True)
                    if is_chatglm:
                        data = (model_output.hidden_states[-1].transpose(0, 1))[0]
                    elif is_t5:
                        data = model_output.encoder_last_hidden_state[0]
                    else:
                        data = model_output.hidden_states[-1][0]
                    data = F.normalize(torch.mean(data, dim=0), p=2, dim=0)
                    embedding.append(data.tolist())
                    token_num += len(input_ids[0])
                ret = {
                    "embedding": embedding,
                    "token_num": token_num,
                }
        except torch.cuda.OutOfMemoryError as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
        except (ValueError, RuntimeError) as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
        return ret


def release_worker_semaphore():
    worker.semaphore.release()


def acquire_worker_semaphore():
    if worker.semaphore is None:
        worker.semaphore = asyncio.Semaphore(worker.limit_worker_concurrency)
    return worker.semaphore.acquire()


def create_background_tasks():
    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_worker_semaphore)
    return background_tasks


@app.post("/worker_generate_stream")
async def api_generate_stream(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    generator = worker.generate_stream_gate(params)
    background_tasks = create_background_tasks()
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_generate")
async def api_generate(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    output = worker.generate_gate(params)
    release_worker_semaphore()
    return JSONResponse(output)


@app.post("/worker_get_embeddings")
async def api_get_embeddings(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    embedding = worker.get_embeddings(params)
    release_worker_semaphore()
    return JSONResponse(content=embedding)


@app.post("/worker_get_status")
async def api_get_status(request: Request):
    return worker.get_status()


@app.post("/count_token")
async def api_count_token(request: Request):
    params = await request.json()
    return worker.count_token(params)


@app.post("/worker_get_conv_template")
async def api_get_conv(request: Request):
    return worker.get_conv_template()


@app.post("/model_details")
async def api_model_details(request: Request):
    return {"context_length": worker.context_len}


if __name__ == "__main__":
    # patch_fastchat()
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    add_model_args(parser)
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument(
        "--limit-worker-concurrency",
        type=int,
        default=5,
        help="Limit the model concurrency to prevent OOM.",
    )
    parser.add_argument("--stream-interval", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    # gptq_config = GptqConfig(
    #     ckpt=args.gptq_ckpt or args.model_path,
    #     wbits=args.gptq_wbits,
    #     groupsize=args.gptq_groupsize,
    #     act_order=args.gptq_act_order,
    # )
    # awq_config = AWQConfig(
    #     ckpt=args.awq_ckpt or args.model_path,
    #     wbits=args.awq_wbits,
    #     groupsize=args.awq_groupsize,
    # )

    worker = BigDLLLMWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.model_path,
        args.model_names,
        args.limit_worker_concurrency,
        no_register=args.no_register,
        device=args.device,
        num_gpus=args.num_gpus,
        max_gpu_memory=args.max_gpu_memory,
        load_8bit=args.load_8bit,
        cpu_offloading=args.cpu_offloading,
        gptq_config=gptq_config,
        awq_config=awq_config,
        stream_interval=args.stream_interval,
        conv_template=args.conv_template,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
