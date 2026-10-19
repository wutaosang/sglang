from types import SimpleNamespace
import requests
from sglang.test.few_shot_gsm8k import run_eval
import os

OUTPUT_DIR = "./profiler_dir"

def _start_profile(**kwargs):
    """Start profiling with optional parameters."""
    requests.post(
        f"http://127.0.0.1:30000/start_profile",
        json=kwargs if kwargs else None,
    )
    # self.assertEqual(response.status_code, 200)

def gsm8k():
    # print(f"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    # _start_profile(start_step="200",num_steps=1)
    # print(f"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    args = SimpleNamespace(
        num_shots=5,
        data_path="/home/z00937177/test.jsonl",
        num_questions=100,   #1319
        max_new_tokens=512,
        parallel=64,
        host="http://127.0.0.1",
        port=8701,
    )
    metrics = run_eval(args)
    print(f"{metrics=}")
    print(f"{metrics['accuracy']=}")
    # self.assertGreater(metrics["accuracy"], 0.7)


if __name__ == "__main__":
    # os.environ["SGLANG_TORCH_PROFILER_DIR"] = OUTPUT_DIR
    gsm8k()
