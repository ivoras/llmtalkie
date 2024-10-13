from __future__ import annotations
from dataclasses import dataclass
import logging
import json
import sys, os
from string import Template
from typing import Any, Callable

import requests


@dataclass(kw_only=True, frozen=True)
class LLMConfig:
    url: str
    model_name: str
    system_message: str
    temperature: float
    options: dict[str,Any]


LLM_LOCAL_LLAMA32 = LLMConfig(
    url = "http://localhost:11434/api/chat",
    model_name = "llama3.2",
    system_message = "You are a knowledgable assistant. You can answer questions and perform tasks.",
    temperature = 0.3,
    options = {
        "num_ctx": 2048,
        "num_predict": -2,
    }
)

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

    DEFAULT_LLM_CONFIG = LLM_LOCAL_LLAMA32

    def __init__(self, *,
                 llm_config: LLMConfig,
                 role: str = "user",
                 callback: Callable[["LLMStep"], dict] = None,
                 previous_step: LLMStep = None,
                 json_response: bool = True,
                 include_history: bool = False,
                 input_data: dict[str, Any],
                 prompt: str):
        self.llm_config = llm_config
        self.role = role
        self.prompt = prompt
        self.callback = callback
        self.previous_step = previous_step
        self.json_response = json_response
        self.include_history = include_history
        self.input_data = input_data

        self.has_response: bool = False
        self.raw_response: str = None # Raw response from the LLM
        self.response: dict = None # Response as dict, parsed from raw_response. If json_response is false, set to {"response": raw_response}
        self.result: dict = None # Whatever the callback returns. If callback is None, set to {"response": raw_response}


class LLMTalkie:
    """
    An agent system for talking to LLMs.
    """

    def __init__(self, *, llm_config: LLMConfig = None, llm_retry: int = 5):
        if llm_config is None:
            llm_config = LLM_LOCAL_LLAMA32
        self.llm_config = llm_config
        self.llm_retry = llm_retry

    def new_step(self, *,
                 llm_config: LLMConfig = None,
                 callback: Callable[["LLMStep"], dict] = None,
                 previous_step: LLMStep = None,
                 include_history: bool = True,
                 json_response: bool = True,
                 input_data: dict[str, Any],
                 prompt: str) -> LLMStep:
        if llm_config is None:
            llm_config = self.llm_config
        return LLMStep(llm_config = llm_config,
                       callback = callback,
                       previous_step = previous_step,
                       include_history = include_history,
                       json_response = json_response,
                       input_data = input_data,
                       prompt = prompt)


    def _count_messages_words(self, messages: str) -> int:
        count = 0
        for m in messages:
            count += m['content'].count(' ')
        return count


    def execute_steps(self, steps: list[LLMStep]):

        def _mk_empty_messages(step: LLMStep):
            if step.llm_config.system_message:
                return [{
                    "role": "system",
                    "content": step.llm_config.system_message,
                }]
            else:
                return []

        for i, step in enumerate(steps):
            if i == 0:
                messages = _mk_empty_messages(step)

            tpl = Template(step.prompt)
            if i > 0:
                step.previous_step = steps[i-1]
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
                messages = _mk_empty_messages(step)

            messages.append({
                "role": step.role,
                "content": content,
            })

            print(f"**** Messages approx word count: {self._count_messages_words(messages)}")

            raw_response: str = None
            response: str|dict = None

            if step.json_response:
                for retry in range(self.llm_retry):
                    r = requests.post(
                        self.llm_config.url,
                        json={
                            "model": step.llm_config.model_name,
                            "options": step.llm_config.options,
                            "stream": False,
                            "keep_alive": "30m",
                            "messages": messages,
                        }
                    )
                    result = r.json()
                    if 'error' in result:
                        log.error(result)
                        break
                    assert result["model"] == step.llm_config.model_name
                    assert result["done"]
                    assert result["message"]["role"] == "assistant"
                    raw_response = result["message"]["content"]
                    try:
                        response = json.loads(raw_response)
                        break
                    except json.JSONDecodeError as e:
                        log.error(f"Error parsing LLM response: {e} ({result['message']['content']}). Try #{retry}/{self.llm_retry}")
                        continue

                if raw_response is None:
                    raise LLMTalkieException("Retry limit reached, LLM did not produce valid JSON")
            else:
                raw_response = result["message"]["content"]
                response = {"response": raw_response}

            step.raw_response = raw_response
            step.response = response
            step.has_response = True
            if step.callback:
                step.result = step.callback(step)
            else:
                step.result = None
