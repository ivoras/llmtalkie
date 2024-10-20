from __future__ import annotations
from dataclasses import dataclass
import logging
import json
import re
import sys, os
from string import Template
from typing import Any, Callable

import requests

RE_MD_JSON = re.compile("```(?:json)?\n(.+)```", re.DOTALL)

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
                 input_callback: Callable[["LLMStep"], dict] = None,
                 validation_callback: Callable[["LLMStep"], bool] = None,
                 result_callback: Callable[["LLMStep"], dict] = None,
                 previous_step: LLMStep = None,
                 json_response: bool = True,
                 include_history: bool = False,
                 input_data: dict[str, Any] = None,
                 prompt: str,
                 trim_prompt: bool = False,
                 ):
        self.llm_config = llm_config
        self.role = role
        self.prompt = prompt
        self.trim_prompt = trim_prompt
        self.input_callback = input_callback
        self.validation_callback = validation_callback
        self.result_callback = result_callback
        self.previous_step = previous_step
        self.json_response = json_response
        self.include_history = include_history
        self.input_data = input_data if input_data is not None else {}

        self.prompt_data: dict = {} # Data passed to the prompt template
        self.has_response: bool = False
        self.raw_response: str = None # Raw response from the LLM
        self.response: dict = None # Response as dict, parsed from raw_response. If json_response is false, set to {"response": raw_response}
        self.result: dict = None # Whatever the callback returns. If callback is None, set to {"response": response}


class LLMTalkie:
    """
    A LLM pipeline orchestrator.
    """

    def __init__(self, *, llm_config: LLMConfig = None, llm_retry: int = 5):
        if llm_config is None:
            llm_config = LLM_LOCAL_LLAMA32
        self.llm_config = llm_config
        self.llm_retry = llm_retry

    def new_step(self, *,
                 llm_config: LLMConfig = None,
                 result_callback: Callable[["LLMStep"], dict] = None,
                 validation_callback: Callable[["LLMStep"], bool] = None,
                 input_callback: Callable[["LLMStep"], dict] = None,
                 previous_step: LLMStep = None,
                 include_history: bool = True,
                 json_response: bool = True,
                 input_data: dict[str, Any] = None,
                 prompt: str,
                 trim_prompt: bool = False) -> LLMStep:
        if llm_config is None:
            llm_config = self.llm_config
        if input_data is None:
            input_data = {}
        return LLMStep(llm_config = llm_config,
                       input_callback = input_callback,
                       validation_callback = validation_callback,
                       result_callback = result_callback,
                       previous_step = previous_step,
                       include_history = include_history,
                       json_response = json_response,
                       input_data = input_data,
                       prompt = prompt,
                       trim_prompt = trim_prompt)


    def _count_messages_words(self, messages: list[dict]) -> int:
        count = 0
        for m in messages:
            count += m['content'].count(' ')
        return count


    def _trim_last_message(self, messages: list[dict], max_words: int):
        prev_messages_len = self._count_messages_words(messages[0:-1])
        message = messages[-1]
        while prev_messages_len + message['content'].count(' ') > max_words:
            p = message['content'].rfind(' ')
            if p == -1:
                break
            message['content'] = message['content'][0:p]


    def execute_steps(self, steps: list[LLMStep]):
        log.debug(f"Executing {len(steps)} steps.")

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

                if step.input_callback:
                    input_data = step.input_callback(step)
                elif step.input_data:
                    input_data = step.input_data
                else:
                    input_data = {}

                step.prompt_data = { **prev_response, **input_data }
                content = tpl.substitute(step.prompt_data)
            else:
                step.prompt_data = step.input_data
                if step.input_data:
                    content = tpl.substitute(step.prompt_data)
                else:
                    content = step.prompt

            if not step.include_history:
                messages = _mk_empty_messages(step)

            messages.append({
                "role": step.role,
                "content": content,
            })

            messages_word_count = self._count_messages_words(messages)
            log.info(f"*** Messages approx word count: {messages_word_count}")
            if messages_word_count > int(step.llm_config.options["num_ctx"] * 0.5) and step.trim_prompt:
                self._trim_last_message(messages, int(step.llm_config.options["num_ctx"] * 0.5))
                messages_word_count = self._count_messages_words(messages)
                log.info(f"    Trimmed to word count: {messages_word_count}")

            step.raw_response = None

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
                    if not 'done' in result or not result['done']:
                        log.error(f"No 'done' field in returned result: {result}")
                        continue
                    assert result["message"]["role"] == "assistant"
                    step.raw_response = self.response = result["message"]["content"]
                    if step.raw_response.find("```") != -1:
                        step.raw_response = RE_MD_JSON.sub(r"\1", step.raw_response)
                    try:
                        step.response = json.loads(step.raw_response)
                        if step.validation_callback:
                            if step.validation_callback(step):
                                break
                            else:
                                log.warning(f"Validation failure. Attempt #{retry}/{self.llm_retry} (response: {step.response})")
                                continue
                        else:
                            break
                    except json.JSONDecodeError as e:
                        log.error(f"Error parsing LLM response: {e} ({result['message']['content']}). Attemp #{retry}/{self.llm_retry}")
                        continue

                if step.raw_response is None:
                    raise LLMTalkieException("Retry limit reached, LLM did not produce valid JSON")

            else:
                # No need to retry prose writing.
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
                step.raw_response = result["message"]["content"]
                self.response = self.result = {"response": step.raw_response}

            step.has_response = True
            if step.result_callback:
                step.result = step.result_callback(step)
            else:
                step.result = {"response": step.raw_response}
