#!/usr/bin/env python3
import re

import requests

import llmtalkie
from wikidig_utils import wikimedia2md

# We'll scrape Wikipedia in search of leaders of big social movements,
# but we'll start from scratch, the "History" page. This is of course
# inefficient, but it will work as a demo. The efficient approach would be
# to use Wikipedia's good old-fashioned search to get more useful starting pages.

def get_page_text(title: str) -> str:
    """
    Returns Markdown-like data for a Wikipedia page with the given title
    """
    title = title.replace(" ", "_")
    url = f"https://en.wikipedia.org/w/index.php?action=raw&title={title}"
    print(url)
    r = requests.get(url)
    text = r.text
    text = wikimedia2md(text)
    return text

def main():

    talkie = llmtalkie.LLMTalkie(
        system_message="You are a helpful research assistent, analyzing topics in Wikipedia articles according to the instructions. You output only JSON documents and nothing else. Do not output explanations or comments.",
    )

    text = get_page_text("History")

    # Step 1 uses a small model with a large context
    step1 = talkie.new_step(
        model="llama3.2",
        options={
            "num_ctx": 16384,
            "temperature": 0.2,
        },
        input_data={"text": text},
        prompt="""
In the following text, list all names of people leading big social movements. If unsure, output an empty list.
Please output data formatted like in this example:

{
  "people": [ "John Doe", "Example Person" ]
}

The text to analyze is:

$text
""".lstrip(),
        callback=lambda x, y: y,
    )

    talkie.execute_steps([step1])

    print(step1.result)


if __name__ == '__main__':
    main()