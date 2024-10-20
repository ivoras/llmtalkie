# llmtalkie - LLM orchestration

A micro LLM agent system for data analysis (or synthesis) pipelines. Currently for OLlama only.

The LLMTalkie project currently provides two features:

1. A data processing pipeline where data can be processed by a sequence of prompts, possibly with a different LLM in each step. It's implemented by the `LLMTalkie` and `LLMStep` classes.
2. A "map" function that applies a prompt (in a single LLM) to a list of data, batching the data efficiently so the LLM can process many items at the same time. It's implemented by the `LLMMap` function.


# The data processing pipeline

You could also call it a "poor man's tool calling for random LLMs".

The idea is to enable workflows like this:

- Take input data
- Adapt it for LLM processing
- Pass it to the first prompt/LLM
- Takt the first LLM's output, adapt the data for the second LLM
- Pass it to the second prompt/LLM
- etc.

The main reason for using multiple prompts is that maybe the data is too complex to process in a single prompt.

The reasons for using different LLMs could be:

- A smaller LLM could be faster to use on a large chunk of data
- A smaller LLM could have a larger context than a bigger one, and can process larger chunks of data
- A bigger LLM could provide deeper insight because of expanded world knowlege, but applied to a distilled chunk of data.

For example (as in the [wikidig](wikidig.py) demo script), a small LLM could fo simpler pre-processing and pass the results to a larger LLM to draw conclusions on.

# The prompt map function

The `LLMMap` function operates on a list of data, processing each element with a LLM. Instead of building item separately into the prompt, it batches input data into chunks and passes it to a LLM as a chunk, increasing efficiency. See the [test_llmupper](test_llmupper.py) example.