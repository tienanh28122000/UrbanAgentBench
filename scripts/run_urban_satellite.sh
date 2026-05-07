#!/bin/bash
AGENT_LLM="openrouter/openai/gpt-5.4"
USER_LLM="openrouter/openai/gpt-5.4"
NUM_TRIALS=1
NUM_TASKS=-1

uab run --domain urban_satellite \
    --agent-llm $AGENT_LLM \
    --user-llm $USER_LLM \
    --num-trials $NUM_TRIALS \
    --num-tasks $NUM_TASKS
