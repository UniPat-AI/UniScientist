import os
import json
import asyncio
import argparse
import traceback
from tqdm import tqdm
from openai import AsyncOpenAI
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional


SYSTEM_PROMPT = """
You are synthesizing multiple research reports produced to solve the same research task into a single, maximally complete and comprehensive final report.

Your goal is to merge all substantive content across the reports into one coherent solution. Do a careful, fine-grained integration:
- Identify and remove redundancy: eliminate duplicated explanations, repeated facts, and overlapping steps while preserving the strongest, clearest phrasing and the most informative details.
- Reconcile differences: when reports disagree, resolve the conflict by choosing the most consistent, well-supported, and internally coherent version. If uncertainty remains, present it transparently as uncertainty and explain how it could be tested or resolved.
- Combine complementary ideas: integrate different approaches, assumptions, heuristics, and methodological choices into a unified, end-to-end method that covers the full problem space.
- Preserve all unique value: do not drop rare but useful edge cases, caveats, failure modes, validation checks, or implementation details that improve correctness or completeness.

Output requirements:
- Produce only the final integrated report. Do not mention, reference, or imply the existence of source reports.
- Organize the report in a structured and logical way with clear headings and a sensible flow.
- Use precise language and provide explicit formulas, definitions, and step-by-step procedures where relevant.
- Prefer actionable specificity over vague summaries. Include concrete parameters, decision rules, and evaluation criteria when available.
- Maintain a consistent terminology and notation throughout the report.
- Preserve full technical detail: keep the complete problem-solving process, all intermediate derivations and equations, and all concrete experimental settings and numeric results used. The report must be logically rigorous and methodologically meticulous rather than high-level or generic.

If the task involves quantitative claims, include the derivations or computation steps needed to reproduce them. If the task involves experiments, specify the experimental setup, controls, metrics, and how to interpret outcomes.
The final integrated report should be as detailed as possible.

Now you can proceed.
""".strip()


def read_jsonl(file_path):
    result = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                result.append(json.loads(line))
    return result


async def call_llm(sem, messages, max_retries=10):
    client = AsyncOpenAI(
        base_url=LOCAL_BASE_URL,
        api_key="",
    )

    async with sem:
        for retry in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model="",
                    messages=messages,
                    temperature=0.6,
                    top_p=0.95,
                    presence_penalty=1.1
                )
                response = response.choices[0].message.model_dump()
                return response
            except Exception as e:
                print(f"Attempt {retry + 1} failed: {e}")
                await asyncio.sleep(1)

    return None


async def call_aggregate(sem, data, messages):
    question = data['question']
    answer = data['answer']
    llm_sem = sem['llm']
    prediction = '[No Prediction]'

    response_message = await call_llm(llm_sem, messages)

    if response_message is None:
        return {'question': question, 'answer': answer, 'rollout': messages, 'prediction': prediction, 'termination': "llm_error_occurred"}

    messages.append(response_message)
    prediction = response_message['content'].split("</think>")[-1].split("<answer>")[-1].split("</answer>")[0].strip()
    termination = 'answer'
    
    return {'question': question, 'answer': answer, 'prediction': prediction, 'rollout': messages, 'termination': termination}


def make_cluster(datasets: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    per_q_counts: Dict[str, List[int]] = defaultdict(lambda: [0] * len(datasets))
    per_q_items: Dict[str, List[Tuple[int, int, Dict[str, Any]]]] = defaultdict(list)
    per_q_answer: Dict[str, Optional[Any]] = {}

    for di, dataset in enumerate(datasets):
        for ri, item in enumerate(dataset):
            q = item.get("question")
            if q is None:
                continue
            per_q_counts[q][di] += 1
            per_q_items[q].append((di, ri, item))

            a = item.get("answer")
            if q not in per_q_answer or per_q_answer[q] is None:
                per_q_answer[q] = a

    clustered_dataset: List[Dict[str, Any]] = []

    for q, items in per_q_items.items():
        k = max(per_q_counts[q])
        if k <= 0:
            continue

        clusters = [{"question": q, "answer": per_q_answer.get(q), "predictions": []} for _ in range(k)]

        items_sorted = sorted(items, key=lambda x: (x[0], x[1]))

        for _, _, item in items_sorted:
            pred = item.get("prediction")
            if pred != '[No Prediction]':
                j = min(range(k), key=lambda idx: len(clusters[idx]["predictions"]))
                clusters[j]["predictions"].append(pred)

        clustered_dataset.extend(clusters)

    return clustered_dataset


async def main(sem, rollout_num, input_paths, output_path):
    datasets = []
    for input_path in input_paths:
        datasets.append(read_jsonl(input_path))
    
    clustered_dataset = make_cluster(datasets)

    visited_dataset = []
    if os.path.exists(output_path):
        checked_dataset = read_jsonl(output_path)
        for visited_data in checked_dataset:
            visited_dataset.append(visited_data['question'])

    # submit task
    tasks = []
    for clustered_data in clustered_dataset:
        if clustered_data['question'] in visited_dataset:
            continue

        question = clustered_data['question']
        predictions = clustered_data['predictions']

        if not predictions:
            continue

        user_prompt = f"[Research Task]\n{question}\n\n"
        for i, prediction in enumerate(predictions):
            user_prompt += f"[Research Report {i+1}]\n{prediction}\n\n"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt.strip()}
        ]
        
        for _ in range(rollout_num):
            tasks.append(call_aggregate(sem, clustered_data, messages))

    print(f"Total number of tasks: {len(tasks)}")

    # process task
    with open(output_path, "a") as f:
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Aggregating ..."):
            try:
                result = await future
    
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())

            except Exception as e:
                exception_type = type(e).__name__
                exception_message = str(e)
                traceback_info = ''.join(traceback.format_tb(e.__traceback__))
                error_message = f'{exception_type}: {exception_message}\n' \
                                f'Traceback:\n{traceback_info}'
                print(f"[ERROR]: {error_message}")
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate multiple rollout results into a single report (test-time scaling)."
    )
    parser.add_argument("--local-base-url", type=str, default="http://localhost:8000/v1",
                        help="vLLM server endpoint (default: http://localhost:8000/v1)")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Model identifier, used for naming the output file")
    parser.add_argument("--data-paths", type=str, nargs="+", required=True,
                        help="One or more input .jsonl files (rollout results from inference_local_qwen.py)")
    parser.add_argument("--benchmark", type=str, default="research",
                        help="Benchmark / task name, used for naming the output file (default: research)")
    parser.add_argument("--rollout-num", type=int, default=1,
                        help="Number of aggregation rollouts per cluster (default: 1)")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Output file path (default: ./<model-name>_<benchmark>_aggregate<N>.jsonl)")
    parser.add_argument("--llm-max-concurrency", type=int, default=32,
                        help="Max concurrent LLM requests (default: 32)")
    args = parser.parse_args()

    sem = {
        'llm': asyncio.Semaphore(args.llm_max_concurrency),
        'tool': asyncio.Semaphore(args.llm_max_concurrency),
    }

    LOCAL_BASE_URL = args.local_base_url

    output_path = args.output_path or f"./{args.model_name}_{args.benchmark}_aggregate{len(args.data_paths)}.jsonl"

    asyncio.run(main(sem, args.rollout_num, args.data_paths, output_path))