"""Prompts for custom summaries"""

# Research article summaries (inspired by: http://ccr.sigcomm.org/online/files/p83-keshavA.pdf)
SYS_MESSAGE_RESEARCH = """
You are a full professor at the prestigious Massachusetts Institute of Technology. You have mastered the skill of reading academic research articles and excell at summarizing their general ideas in simple English.
"""

MAP_PROMPT_RESEARCH = """
Article Page:
```{text}```

Instructions:
Your task is to generate a summary of the provided `Article Page` of an academic research article.
Summarize the provided page, delimited by triple backticks, in at most {max_words} words.
Be detailed and specific when writting your page summary.

Page Summary:
"""


REDUCE_PROMPT_RESEARCH = """
Page Summaries:
```
{text}
```

Output Format:
# Title

## Category:
[What type of paper is this? An empirical paper? An analysis of an existing system? A description of a research prototype? A theoretical paper?]

## Context:
[Which other specific papers is it related to? Which theoretical bases were used to analyze the problem?]

## Correctness:
[Do the assumptions appear to be valid? Appear any of the assumptions to be problematic?]

## Contributions:
[What are the paper's main contributions? What is the novelty?]

## Clarity:
[Is the paper well written?]

Instructions:
You are provided with `Page Summaries` of an academic research article delimited by triple back ticks. Your task is to write a final, coherent summary of the article in at most {max_words} words.
The final summary must answer the five C's (Category, Context, Correctness, Contributions, Clarity) in the provided `Output Format`.
Be detailed and specific when writting your final summary. Remember, to only provide the final summary according to the `Output Format`.
"""
