from __future__ import annotations
import logging
import json
import sys, os
from string import Template
from typing import Any, Callable

import requests

class LLMTalkieException(Exception): pass

# Design requirements:
#
# Inputs are text prompts, outputs always JSON
# Each conversation step could be done by a different LLM
#
# Example:
# talkie = LLMTalkie(...defaults..., system_message="...")
# step1 = LLMStep(...gemma:2b..., big text, extract some information from this text into JSON, callback to parse this json into dict)
# step2 = LLMStep(...qwen2.5..., data step1 templated with some template, Python format_map from prompt, "Extract data from {title}")
#
# step3 =

log = logging.getLogger("llmtalkie")

class LLMStep:
    """
    One step in the LLM chat turn, consisting of a user step and its response (an assistent step).
    Construct one of these for every message in the chat history / plan.
    """

    DEFAULT_URL = "http://localhost:11434/api/chat"
    DEFAULT_MODEL = "llama3.2"

    DEFAULT_OPTIONS = {
        "num_ctx": 2048,
        "num_predict": -2
    }

    def __init__(self, *,
                 url: str = DEFAULT_URL,
                 model: str = DEFAULT_MODEL,
                 options: dict[str, Any] = DEFAULT_OPTIONS,
                 role: str = "user",
                 callback: Callable[["LLMStep", str|dict], dict] = None,
                 previous_step: LLMStep = None,
                 json_response: bool = True,
                 include_history: bool = False,
                 input_data: dict[str, Any],
                 prompt: str):
        self.url = url
        self.model = model
        self.options = options
        self.role = role
        self.prompt = prompt
        self.callback = callback
        self.previous_step = previous_step
        self.json_response = json_response
        self.include_history = include_history
        self.input_data = input_data

        self.has_response = False
        self.raw_response = None # Raw response from the LLM
        self.result: dict = None # Whatever the callback returns. If callback is None, set to {"response": raw_response}


class LLMTalkie:
    """
    An agent system for talking to LLMs.
    """

    def __init__(self, *, url: str = LLMStep.DEFAULT_URL, model: str = LLMStep.DEFAULT_MODEL, options: dict[str, Any] = LLMStep.DEFAULT_OPTIONS, system_message: str = None, llm_retry: int = 5):
        self.url = url
        self.model = model
        self.options = options
        self.system_message = system_message
        self.llm_retry = llm_retry

    def new_step(self, *,
                 url: str = None,
                 model: str = None,
                 options: dict[str, Any] = None,
                 callback: Callable[["LLMStep", str|dict], dict] = None,
                 previous_step: LLMStep = None,
                 include_history: bool = True,
                 json_response: bool = True,
                 input_data: dict[str, Any],
                 prompt: str) -> LLMStep:
        if url is None:
            url = self.url
        if model is None:
            model = self.model
        if options is None:
            options = self.options
        return LLMStep(url = url,
                       model = model,
                       options = options,
                       callback = callback,
                       previous_step = previous_step,
                       include_history = include_history,
                       json_response = json_response,
                       input_data = input_data,
                       prompt = prompt)


    def _mk_empty_messages(self):
        if self.system_message:
            return [{
                "role": "system",
                "content": self.system_message,
            }]
        else:
            return []

    def _count_messages_words(self, messages: str) -> int:
        count = 0
        for m in messages:
            count += m['content'].count(' ')
        return count


    def execute_steps(self, steps: list[LLMStep]):
        messages = self._mk_empty_messages()
        for i, step in enumerate(steps):
            tpl = Template(step.prompt)
            if step.previous_step:
                if i == 0:
                    raise LLMTalkieException(f"First step cannot have previous_step")
                prev_response = step.previous_step.result if step.previous_step.result else {}
                input_data = step.input_data if step.input_data else {}
                prompt_data = { **prev_response, **input_data }
                content = tpl.substitute(prompt_data)
            else:
                if step.input_data:
                    content = tpl.substitute(step.input_data)
                else:
                    content = step.prompt

            if not step.include_history:
                messages = self._mk_empty_messages()

            messages.append({
                "role": step.role,
                "content": content,
            })

            print(f"**** Messages approx word count: {self._count_messages_words(messages)}")

            raw_response: str|dict = None
            if step.json_response:
                for retry in range(self.llm_retry):
                    r = requests.post(
                        self.url,
                        json={
                            "model": step.model,
                            "options": step.options,
                            "stream": False,
                            "keep_alive": "30m",
                            "messages": messages,
                        }
                    )
                    result = r.json()
                    print(result)
                    assert result["model"] == step.model
                    assert result["done"]
                    assert result["message"]["role"] == "assistant"
                    try:
                        raw_response = json.loads(result["message"]["content"])
                        break
                    except json.JSONDecodeError as e:
                        log.error(f"Error parsing LLM response: {e} ({result['message']['content']}). Try #{retry}/{self.llm_retry}")
                        continue

                if raw_response is None:
                    raise LLMTalkieException("Retry limit reached, LLM did not produce valid JSON")
            else:
                raw_response = result["message"]["content"]

            step.raw_response = raw_response
            step.has_response = True
            if step.callback:
                step.result = step.callback(step, raw_response)

