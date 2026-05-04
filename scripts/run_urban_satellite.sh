#!/bin/bash
AGENT_LLM="openrouter/z-ai/glm-4.5v" # gpt-4.1-mini, gpt-5-mini, gpt-4.1-nano, gpt-5-nano
USER_LLM="openrouter/openai/gpt-4.1-mini"
NUM_TRIALS=1
NUM_TASKS=100

uab run --domain urban_satellite \
    --agent-llm $AGENT_LLM \
    --user-llm $USER_LLM \
    --num-trials $NUM_TRIALS \
    --num-tasks $NUM_TASKS
