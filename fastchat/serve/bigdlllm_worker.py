"""
A model worker that executes the model based on BigDL-LLM.

See documentations at docs/bigdlllm_integration.md
"""

import argparse
import asyncio
import dataclasses
import logging
import json
import os
import time
from typing import List, Optional
import threading
import uuid
from bigdl.llm.utils.common import invalidInputError
from bigdl.llm.utils import load_model

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse

from fastchat.modules.gptq import GptqConfig
from fastchat.modules.awq import AWQConfig
from fastchat.serve.base_model_worker import BaseModelWorker
from fastchat.serve.model_worker import (
    logger,
    worker_id,
)
import requests

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
        load_8bit: bool = False,
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
        from fastchat.model.model_adapter import load_model
        self.model, self.tokenizer = load_model(
            model_path,
            device=device,
            num_gpus=num_gpus,
            max_gpu_memory=max_gpu_memory,
            load_8bit=load_8bit,
            cpu_offloading=cpu_offloading,
            gptq_config=gptq_config,
            awq_config=awq_config,
        )