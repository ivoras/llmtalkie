#!/usr/bin/env python3
import re
from time import sleep

import pprint
import requests

from llmtalkie import LLMTalkie, LLMStep
from wikidig_utils import wikimedia2md

# We'll scrape Wikipedia in search of political leaders,
# but we'll start from scratch, the "History" page. This is of course
# inefficient, but it will work as a demo. The efficient approach would be
# to use Wikipedia's good old-fashioned search to get more useful starting pages.
# and extract mechanical data like links with regexps instead of LLMs.
#
# This demo requires a system with at least 12 GB of VRAM and
# the following LLMs to be installed in ollama:
#
# * llama3.2
# * qwen2.5:14b

RE_WP_REDIRECT = re.compile(r"#REDIRECT\s*\[(.+?)[\n\]]")

def get_page_text(title: str) -> str:
    """
    Returns Markdown-like data for a Wikipedia page with the given title
    """
    title = title.replace(" ", "_")
    url = f"https://en.wikipedia.org/w/index.php?action=raw&title={title}"
    r = requests.get(url)
    sleep(0.1) # take it easy on Wikipedia
    return wikimedia2md(r.text)


def main():

    people_names_queue = []
    pages_queue = []

    result = {} # e.g. { "Person Name": "What they did" }

    talkie = LLMTalkie(
        system_message="You are a helpful research assistent, analyzing topics in Wikipedia articles according to the instructions. You output only JSON documents and nothing else. Do not output explanations or comments.",
    )

    pages_queue.append("History") # Starting page

    text = get_page_text(pages_queue.pop())

    def fetch_people_descriptions(step: LLMStep):
        assert type(step.response) == dict
        people_sections = []
        for name in step.response["people"]:
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

        return {"people": "\n\n".join(people_sections)} # gets into step2's prompt via template


    # Step 1 uses a small and fast model
    step1 = talkie.new_step(
        model="llama3.2",
        options={
            "num_ctx": 16384,
            "temperature": 0.2,
        },
        input_data={"text": text},
        prompt="""
List up to 30 people names appearing in the following text. If unsure, output an empty list.
Please output data formatted as JSON like in this example:

{
  "people": [ "John Doe", "Person Name" ]
}

The text to analyze is:

$text
""".lstrip(),
        # Called when LLM generates the response from this step, prepares the result for the following step
        callback=fetch_people_descriptions,
    )

    step2 = talkie.new_step(
        model="qwen2.5:14b", # we need a smarter model for this step
        options={
            "num_ctx": 16384,
            "temperature": 0.2,
        },
        input_data={"text": text},
        prompt="""
Your task is to analyze the following sections containing texts about people. Each section is
titled with a person's name. Analyze all the available sections and for each person,
output YES or NO, depending of if the text for the person indicates
they were a political leader.

Please output the data formatted as JSON like in this example:

{
    "people": {
        "Person name": "YES",
        "Another person name": "NO"
    }
}

The text to analyze is:

$people
""".lstrip(),
        callback=lambda x: x,
    )


    talkie.execute_steps([step1, step2])

    #pprint.pp(step1.result)
    pprint.pp(step2.response)


if __name__ == '__main__':
    main()