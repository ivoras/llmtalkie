#!/usr/bin/env python3
import re
from time import sleep

import pprint
import requests
import feedparser

from llmtalkie import LLMTalkie, LLMStep, LLMConfig

# Why 2 LLMs? The first one knows the Bible, the second one knows how to write good sermons ;)

LLM_LOCAL_COMMAND_R = LLMConfig(
    url = "http://localhost:11434/api/chat",
    model_name = "command-r:35b-08-2024-q4_K_S",
    system_message = "You are a helpful research, religious assistent of the Roman Catholic faith, analyzing news topics. You output only JSON documents and nothing else. Do not output explanations or comments. Stop after outputting JSON.",
    temperature = 0.2,
    options = {
        "num_ctx": 2048, # Almost every time the LLM returns prose instead of JSON, the context size is too small
        "num_predict": -2,
    }
)

LLM_LOCAL_DARKEST_PLANET = LLMConfig(
    url = "http://localhost:11434/api/chat",
    model_name = "darkest-planet",
    system_message = "You are a prophet of the Lord, of the Roman Catholic faith. You write sermons that inspire people to change their lives and participate in solving big world problems. Your sermons are long and sometimes reference Bible verses, your optimism and faith in the Lord and in the betterment of humanity are contagious.",
    temperature = 0.8,
    options = {
        "num_ctx": 4096,
        "num_predict": -2,
    }
)

def main():
    talkie = LLMTalkie()

    # Get World News
    d = feedparser.parse("https://feeds.bbci.co.uk/news/world/rss.xml")
    # Ask the LLM to generate Bible references relating to world news
    step1 = LLMStep(
        llm_config = LLM_LOCAL_COMMAND_R,
        input_data = {"news_items": "\n".join(f"* {entry.title}. {entry.description}" for entry in d.entries) },
        prompt = """
In the following news items, find 3 topics that are of greates concern to pious Catholics, having an impact on their duties, beliefs or institutions. For each of those topics, find a Bible reference that is most suitable for the topic.

Output just a JSON document in the following format:

{
    "topics": [
        {
            "topic": "Topic name",
            "news_item": "The news item text",
            "bible_reference: "Bible reference, for example Ezekiel 25:17",
            "bible_quote": "Bible quote for the above reference."
        }
    ]
}

# News items

$news_items
""".lstrip()
    )

    step2 = LLMStep(
        llm_config = LLM_LOCAL_DARKEST_PLANET,
        input_callback = lambda step: { "topic_sections": "\n".join([f"## {t['topic']}\n\n* Bible reference: {t['bible_reference']}\n* Bible quote: {t['bible_quote']}\n* Relation to the world events: {t['news_item']}\n" for t in step.previous_step.response['topics']]) },
        prompt = """
Write a sermon worthy of the Pope on the impact of the topics described in the following sections on the modern world.

Use Bible quotes sparingly. Envision a world when, with Jesus's help the described problems will be solved, through the workings of good men. Also mention one of the general problems of the Church in modern times and call on all the faithful to aid in solving it. The sermon should be long, and it should inspire listeners to act for the betterment of the global society. It should also contain dire apocalyptic warnings of what will happen if the poeople do not contribute to a better world.

Do not reference Bible quotes that are not included in the sections below. Do not output commentary of the sermon, or instructions of what the faithful should do during the mass. Begin with "Blessings of peace to all of you, my brothers and sisters." Finish with "Amen."

$topic_sections

""".lstrip(),
    json_response = False,
    )

    talkie.execute_steps([step1, step2])

    pprint.pp(step1.response)
    pprint.pp(step2.response, width=120)






if __name__ == '__main__':
    main()