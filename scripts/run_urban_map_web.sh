#!/bin/bash

# Define the user model (usually kept constant for consistency)
USER_LLM="openrouter/openai/gpt-5.4"

# List of agent models to evaluate
AGENT_MODELS=(
    ## flagship models
    # "openrouter/openai/gpt-5.4"
    # "openrouter/anthropic/claude-sonnet-4.6"
    # "openrouter/anthropic/claude-opus-4.6"
    # "openrouter/google/gemini-3.1-pro-preview"
    ## closed-source models
    "openrouter/openai/gpt-5.4-mini"
    # "openrouter/google/gemini-3-flash-preview"
    ## open-source models
    # "openrouter/qwen/qwen3.5-397b-a17b"
    # "openrouter/deepseek/deepseek-v3.2"
    # "openrouter/z-ai/glm-5"
    # "openrouter/moonshotai/kimi-k2.5"
)

# Configuration
NUM_TRIALS=1
NUM_TASKS=-1
DOMAIN="urban_map_web"

# Loop through each agent model
for AGENT_LLM in "${AGENT_MODELS[@]}"; do
    echo "======================================================"
    echo "Running evaluation for Agent: $AGENT_LLM"
    echo "======================================================"

    uab run --domain $DOMAIN \
        --agent-llm "$AGENT_LLM" \
        --agent-llm-args '{"max_tokens":2048}' \
        --user-llm "$USER_LLM" \
        --user-llm-args '{"max_tokens":2048}' \
        --num-trials $NUM_TRIALS \
        --num-tasks $NUM_TASKS

    echo "Finished $AGENT_LLM"
done

echo "All evaluations completed."
