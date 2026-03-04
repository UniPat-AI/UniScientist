import os
import json
import copy
import json5
import asyncio
import traceback
from tqdm import tqdm
from openai import AsyncOpenAI
from collections import Counter

from tools.tool_search import Search
from tools.tool_visit import Visit
from tools.tool_scholar import ScholarSearch
from tools.tool_code import PythonInterpreter


LOCAL_BASE_URL = os.getenv("LOCAL_BASE_URL")


SYSTEM_PROMPT = """
You are an AI scientist. Your job is to solve the user's problem as accurately as possible.
Use the provided tools wisely to gather missing or uncertain information, verify claims, and obtain relevant evidence. Prefer tool calls when they materially improve correctness, completeness, or confidence.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "search", "description": "Perform Google web searches then returns a string of the top search results. Accepts multiple queries.", "parameters": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string", "description": "The search query."}, "minItems": 1, "description": "The list of search queries."}}, "required": ["query"]}}}
{"type": "function", "function": {"name": "visit", "description": "Visit webpage(s) and return the summary of the content.", "parameters": {"type": "object", "properties": {"url": {"type": "array", "items": {"type": "string"}, "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."}, "goal": {"type": "string", "description": "The specific information goal for visiting webpage(s)."}}, "required": ["url", "goal"]}}}
{"type": "function", "function": {"name": "google_scholar", "description": "Leverage Google Scholar to retrieve relevant information from academic publications. Accepts multiple queries. This tool will also return results from google search", "parameters": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string", "description": "The search query."}, "minItems": 1, "description": "The list of search queries for Google Scholar."}}, "required": ["query"]}}}
{"type": "function", "function": {"name": "python_interpreter", "description": "Execute Python code in a sandboxed environment and return the execution results.\n\nKey Points:\n- Use print() statements for any output you want to see\n- Only Python standard library is available (e.g., math, json, re, datetime, collections, statistics, itertools, functools, etc.)\n- DO NOT create, modify, or delete files; keep all operations in-memory (no file I/O)\n- DO NOT install or download any external packages or resources\n- Focus on computation, data processing, and calculations\n- IMPORTANT: Each call is completely stateless. Treat every execution as a fresh interpreter with an empty namespace. Never reuse or depend on anything from previous calls.", "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Provide the Python code to execute as ONE single string for THIS CALL ONLY.\n\nHard rules (non-negotiable):\n- Execute exactly and only the code in this field for the current call. Do NOT prepend, append, merge, or reuse any code from previous turns.\n- Assume a brand-new Python interpreter every call: empty namespace, no prior variables/functions/imports/state/output exist.\n- The code must be fully self-contained and runnable on its own.\n- OUTPUT IS PRINT-ONLY: The tool returns ONLY stdout produced by print(). There is NO REPL echo, NO implicit expression output, and NO visible return values. If you want any value, result, intermediate, or final output to be seen, you MUST explicitly print it with print(). Anything not printed is treated as invisible.\n- No file I/O (no file creation/modification/deletion). Keep everything in-memory.\n- Do not install, import, or download any non-standard-library packages or any external resources."}}, "required": ["code"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
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
                if not response:
                    raise ValueError("LLM generation error occurred unexpectedly.")
                response = response.choices[0].message.model_dump()
                return response
            except Exception as e:
                print(f"Attempt {retry + 1} failed: {e}")
                await asyncio.sleep(1)

    return None


async def call_tool(sem, tool_name: str, tool_args: dict):
    async with sem:
        if tool_name == "search":
            return await search.call(tool_args)
        elif tool_name == "visit":
            return await visit.call(tool_args)
        elif tool_name == "google_scholar":
            return await scholar.call(tool_args)
        elif tool_name == "python_interpreter":
            await asyncio.sleep(0.01)
            return interpreter.call(tool_args)
        else:
            await asyncio.sleep(1)
            return f'Tool {tool_name} does not exist.'


async def agent_rollout(sem, data, messages, max_turn=100):
    question = data['problem']
    answer = data['answer']

    llm_sem = sem['llm']
    tool_sem = sem['tool']
    
    record = copy.deepcopy(messages)

    termination = 'max_turn_exceeded'
    prediction = '[No Prediction]'

    for turn in range(max_turn):

        response_message = await call_llm(llm_sem, record)

        if response_message is None:
            return {'question': question, 'answer': answer, 'rollout': record, 'termination': "llm_error_occurred"}

        llm_response = response_message['content']
        record.append(response_message)
        
        if "<tool_call>" in llm_response and "</tool_call>" in llm_response:
            tool_call_str = llm_response.split('<tool_call>')[-1].split('</tool_call>')[0]

            try:
                tool_call = json5.loads(tool_call_str)

                tool_name = tool_call['name']
                tool_args = tool_call['arguments']
                
                if isinstance(tool_args, str):
                    tool_args = json5.loads(tool_args)

                tool_response = await call_tool(tool_sem, tool_name, tool_args)

                print("======================================")
                print(f"Call `{tool_name}`: {tool_args}")
                print(f"Tool call {tool_name} invocation success with length {len(tool_response)}")
                print(tool_response)
            except Exception as e:
                tool_response = 'Error: Tool call is not a valid JSON. Tool call must contain a valid "name" and "arguments" field.'
                print(f"Tool call invocation error {e}")

            record.append({"role": "user", "content": f"<tool_response>\n{tool_response}\n</tool_response>"})

        else:
            prediction = llm_response.split("</think>")[-1].split("<answer>")[-1].split("</answer>")[0].strip()
            termination = 'answer'
            break
            
    return {'question': question, 'answer': answer, 'prediction': prediction, 'rollout': record, 'termination': termination}


async def main(sem, rollout_count, input_path, output_path):
    dataset = read_jsonl(input_path)
    
    visited_counter = Counter()
    if os.path.exists(output_path):
        existing_rollouts = read_jsonl(output_path)
        for visited_data in existing_rollouts:
            question = visited_data['question']
            visited_counter[question] += 1

    # submit task
    tasks = []
    pending_counter = Counter()
    for data in dataset:
        question = data.get('problem')
        target = rollout_count

        total_count = visited_counter[question] + pending_counter[question]
        need_to_submit = max(0, target - total_count)

        for _ in range(need_to_submit):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}
            ]
            tasks.append(agent_rollout(sem, data, messages))
            pending_counter[question] += 1

    print(f"Total number of tasks: {len(tasks)}")

    # process task
    with open(output_path, "a") as f:
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Rollout ..."):
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
    sem = {
        'llm': asyncio.Semaphore(int(os.getenv("LLM_MAX_CONCURRENCY"))),
        'tool': asyncio.Semaphore(int(os.getenv("TOOL_MAX_CONCURRENCY"))),
    }

    benchmark = os.getenv("BENCHMARK")
    stored_model_name = os.getenv("STORED_MODEL_NAME")
    rollout_count = int(os.getenv("ROLLOUT_COUNT"))

    data_path = os.getenv("DATA_PATH", "")
    if not data_path:
        raise ValueError("DATA_PATH environment variable is required.")
    output_path = f"./{stored_model_name}_{benchmark}.jsonl"

    search = Search()
    visit = Visit()
    scholar = ScholarSearch()
    interpreter = PythonInterpreter(timeout=120)

    asyncio.run(main(sem, rollout_count, data_path, output_path))
