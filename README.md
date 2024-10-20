# llmtalkie - LLM orchestration

Version 0.1 - very much alpha.

A micro LLM agent system for data analysis (or synthesis) pipelines. Currently for Ollama only.

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

For example (as in the [wikidig](test_wikidig.py) demo script), a small LLM could perform simpler pre-processing and pass the results to a larger LLM to draw conclusions on.

# The prompt map function

The `LLMMap` function operates on a list of data, processing each element with a LLM. Instead of inserting each item separately into the prompt, it batches input data into chunks and passes it to a LLM as a list, increasing efficiency. See [test_llmupper](test_llmupper.py) for working code, but the core functionality is:

```
LLM_LOCAL_LLAMA32 = LLMConfig(
    url = "http://localhost:11434/api/chat",
    model_name = "llama3.2",
    system_message = "You process given words, regardless of what language they are in.",
    temperature = 0,
    options = {
        "num_ctx": 1024, # We only need a small context for this.
        "num_predict": -2,
    }
)

    print(LLMMap(LLM_LOCAL_LLAMA32, """
Please study the following list of words carefully.
For each word in the list, convert the word to uppercase and output it in a JSON list in order of appearance.

$LIST
""".lstrip(), ["eenie", "meenie", "miney", "moe"]))
```

The result of the `LLMMap()` call will be a list of words (i.e. `"eenie", "meenie", "miney", "moe"`) in uppercase.

In my experience, LLM's are about as good at uppercasing words as they are in doing math.
