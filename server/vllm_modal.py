"""Serve Qwen3-8B with vLLM on one Modal L4 GPU.

Adapted from Modal's example 06_gpu_and_ml/llm-serving/vllm_inference.py.

Deploy:   uv run modal deploy server/vllm_modal.py
Smoke:    uv run modal run server/vllm_modal.py
Logs:     uv run modal app logs vllm-serving-lab
Stop:     uv run modal app stop vllm-serving-lab

Needs a Modal secret named "vllm-api-key" holding VLLM_API_KEY; the same key
must be in the local environment for the smoke test and the clients.
"""

import os
import time

import modal

MODEL_NAME = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"  # pin so reruns use the same weights
GPU = "L4"
VLLM_PORT = 8000
MINUTES = 60

# Engine settings under study; change one at a time for the experiments.
ENGINE_ARGS = [
    "--max-model-len", "8192",  # the default 40,960 does not fit next to bf16 weights on 24 GB
    "--gpu-memory-utilization", "0.90",
]

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "VLLM_LOG_STATS_INTERVAL": "1"})
)

# Cache weights and compiled kernels so restarts don't download or compile again.
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("vllm-serving-lab")


@app.server(
    image=vllm_image,
    gpu=GPU,
    secrets=[modal.Secret.from_name("vllm-api-key")],
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    port=VLLM_PORT,
    max_containers=1,  # one GPU only, so benchmarks measure a single vLLM engine
    scaledown_window=15 * MINUTES,
    startup_timeout=20 * MINUTES,  # the first start downloads about 16 GB
    unauthenticated=True,  # Modal's proxy is open; vLLM checks the API key
)
class Server:
    @modal.enter()
    def start(self):
        import subprocess

        cmd = [
            "vllm", "serve", MODEL_NAME,
            "--revision", MODEL_REVISION,
            "--host", "0.0.0.0",
            "--port", str(VLLM_PORT),
            *ENGINE_ARGS,
        ]
        # No --api-key flag: vLLM logs its non-default CLI args, which would print the key.
        # It reads VLLM_API_KEY from the environment instead (set by the Modal secret).
        assert os.environ.get("VLLM_API_KEY"), "secret vllm-api-key is missing"
        print(*cmd)
        self.process = subprocess.Popen(cmd)

    @modal.exit()
    def stop(self):
        self.process.terminate()


@app.local_entrypoint()
def smoke_test():
    """Wait for /health, then list the models with the API key."""
    import httpx

    url = Server.get_url()
    headers = {"Authorization": f"Bearer {os.environ['VLLM_API_KEY']}"}
    deadline = time.time() + 20 * MINUTES
    with httpx.Client(base_url=url, timeout=60) as client:
        while time.time() < deadline:
            try:
                if client.get("/health").status_code == 200:
                    break
            except httpx.TransportError:  # the proxy can time out while the container starts
                pass
            time.sleep(5)
        else:
            raise TimeoutError(f"{url} did not become healthy")
        response = client.get("/v1/models", headers=headers)
        response.raise_for_status()
        print(url)
        print(response.json())
