#!/usr/bin/env python3
from collections import deque
import json
import logging
import re
import sys
from time import sleep

import pprint
import requests

from llmtalkie import LLMTalkie, LLMStep, LLMConfig
from wikidig_utils import wikimedia2md

log = logging.getLogger("llmtalkie")
logging.basicConfig(stream=sys.stderr, level=logging.INFO)

# This is a demo app for llmtalkie.
# We'll scrape Wikipedia in search of political leaders,
# but we'll start from scratch, the "History" page. This is of course
# inefficient, but it will work as a demo. The efficient approach would be
# to use Wikipedia's good old-fashioned search to get more useful starting pages.
# and extract mechanical data like links with regexps instead of LLMs.
#
# This demo requires a system with at least 24 GB of VRAM (or RAM if you're patient) and
# the following LLMs to be installed in ollama:
#
# * llama3.2
# * qwen2.5:14b
#
# The idea is to use llama3.2, a small and fast model that allows us to have a bigger context,
# for simpler tasks, and then defer to qwen2.5-14b, a much bigger model which only
# allows us a smaller context, to draw conclusions.

LLM_LOCAL_LLAMA32 = LLMConfig(
    url = "http://localhost:11434/api/chat",
    model_name = "llama3.2",
    system_message = "You are a helpful research assistent, analyzing topics in Wikipedia articles according to the instructions. You output only JSON documents and nothing else. Do not output explanations or comments. Stop after outputting JSON.",
    temperature = 0.3,
    options = {
        "num_ctx": 16384, # Almost every time the LLM returns prose instead of JSON, the context size is too small
        "num_predict": -2,
    }
)

LLM_LOCAL_QWEN25_14B = LLMConfig(
    url = "http://localhost:11434/api/chat",
    model_name = "qwen2.5:14b",
    system_message = "You are a helpful research assistent, analyzing topics in Wikipedia articles according to the instructions. You output only JSON documents and nothing else. Do not output explanations or comments. Stop after outputting JSON.",
    temperature = 0.2,
    options = {
        "num_ctx": 6144,
        "num_predict": -2,
    }
)

RE_WP_REDIRECT = re.compile(r"#REDIRECT\s*\[(.+?)[\n\]]")

def get_page_text(title: str) -> str:
    """
    Returns Markdown-like data for a Wikipedia page with the given title
    """
    title = title.replace(" ", "_")
    url = f"https://en.wikipedia.org/w/index.php?action=raw&title={title}"
    r = requests.get(url)
    sleep(0.1) # take it easy on Wikipedia
    text = wikimedia2md(r.text)
    text = text.replace("'''", "")
    text = text.replace("''", "")
    return text


def main():
    pages_queue = deque()
    pages_queue.append("History") # Starting page
    seen_pages: set[str] = set(["history"])

    result = {} # e.g. { "Person Name": "What they did" }

    talkie = LLMTalkie()
    count_pages = 0

    def update_pages_queue(step: LLMStep):
        if type(step.response) != dict:
            print(type(step.response))
            print(step.response)
        if not step.response:
            return
        for page in step.response["pages"]:
            if page.lower() in seen_pages:
                continue
            seen_pages.add(page.lower())
            pages_queue.append(page)
        return step.response

    def fetch_people_descriptions(step: LLMStep):
        if type(step.response) != dict:
            print(type(step.response))
            print(step.response)
        if not step.response:
            return {"people": "None", "people_descriptions": []}
        people_sections = []
        people_descriptions = {}
        for name in step.response["people"]:
            if name in people_descriptions:
                continue
            text = get_page_text(name)

            while text.find("#REDIRECT") != -1:
                m = RE_WP_REDIRECT.search(text)
                if m:
                    text = get_page_text(m.group(1))
                else:
                    text = "Wikimedia Error"
                    break

            if text.find("Wikimedia Error") != -1:
                continue

            text = text[0:text.find("\n#")] # get the first section of the Wikipedia page, the one with the description
            people_sections.append(f"# {name}\n\n{text}\n")
            people_descriptions[name] = text

        return {"people": "\n\n".join(people_sections), "people_descriptions": people_descriptions} # gets into step2's prompt via template

    def update_result(step: LLMStep):
        if type(step.response) != dict:
            print(type(step.response))
            print(step.response)
        if not step.response:
            return
        if not 'people' in step.response:
            print("ERROR: no people in:", step.response)
            return
        for name in step.response["people"]:
            if step.response["people"][name].upper() == "YES":
                try:
                    description = step.previous_step.result["people_descriptions"][name]
                    result[name] = description
                    print("Found person:", name)
                except KeyError:
                    print(f"Can't find person {name} in previous_step inputs: {step.previous_step.input_data}")

    while len(pages_queue) > 0 and count_pages < 50: # Process up to 50 pages
        page_title = pages_queue.popleft()
        print("Processing page:", page_title)
        text = get_page_text(page_title)

        # Step 1 and 2 use a small and fast model
        step1 = talkie.new_step(
            llm_config=LLM_LOCAL_LLAMA32,
            input_data={"text": text},
            trim_prompt=True,
            prompt="""
You are given a text from a Wikipedia article in the section titled "Text" and it has links to other pages formatted as "[Page title]".
Please list page titles about political events or social movements.
Please write the data formatted as a JSON document like in this example:

{
    "pages": [ "Page title", "Another page title" ]
}

If not sure, make the "pages" list empty.

# Text

$text
""".lstrip(),
            validation_callback=lambda step: type(step.response) == dict and 'pages' in step.response,
            # The LLM result is structured just the way we want it, we just want to fill the page_queue with this callback.
            result_callback=update_pages_queue,
        )

        step2 = talkie.new_step(
            llm_config=LLM_LOCAL_LLAMA32,
            input_data={"text": text},
            trim_prompt=True,
            prompt="""
List up to 30 people names appearing in section titled "Text", who might be involved in politics or social movements.
Please write the data formatted as a JSON document like in this example:

{
    "people": [ "John Doe", "Person Name" ]
}

If not sure, make the "people" list empty.

# Text

$text
""".lstrip(),
            include_history=False,
            validation_callback=lambda step: type(step.response) == dict and 'people' in step.response,
            # Called when LLM generates the response from this step, prepares the result for the following step
            result_callback=fetch_people_descriptions,
        )

        step3 = talkie.new_step(
            llm_config=LLM_LOCAL_QWEN25_14B,
            input_data={"text": text},
            include_history=False,
            trim_prompt=True,
            prompt="""
Your task is to analyze the following sections containing texts about people. Each section is
titled with a person's name. Analyze all the available sections and for each person,
output "YES" if the text for the person indicates they were either a political leader,
social movement leader, or a revolutionary, and otherwise wrie "NO".

Please output the data formatted as JSON like in this example, without any commentary or explanations:

{
    "people": {
        "Person name": "YES",
        "Another person name": "NO"
    }
}

The texts to analyze are:

$people
""".lstrip(),
            validation_callback=lambda step: type(step.response) == dict and 'people' in step.response,
            result_callback=update_result,
        )

        talkie.execute_steps([step1, step2, step3])

        #pprint.pp(step1.result, width=120)
        #pprint.pp(step2.result, width=120)
        #pprint.pp(step3.result, width=120)
        print("# Pages to process:", len(pages_queue))

    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
