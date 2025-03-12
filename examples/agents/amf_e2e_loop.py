# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import asyncio
import os
import logging
import fire
from examples.client_tools.ticker_data import get_ticker_data
from examples.client_tools.web_search import WebSearchTool
from examples.client_tools.calculator import calculator
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig
from termcolor import cprint

# Turn down the httpx logging
# Get the logger for 'httpx'
http_logger = logging.getLogger("httpcore")
# Set the logging level to WARNING
http_logger.setLevel(logging.WARNING)

# Get the logger for the llama stack client
http_logger = logging.getLogger("llama_stack_client")
# Set the logging level to WARNING
http_logger.setLevel(logging.WARNING)

# Set up logging for myself
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_main(host: str, port: int, disable_safety: bool = False):
    if "TAVILY_SEARCH_API_KEY" not in os.environ:
        print(
            colored(
                "Warning: TAVILY_SEARCH_API_KEY is not set; will not use websearch tool.",
                "yellow",
            )
        )
    
    client = LlamaStackClient(
        base_url=f"http://{host}:{port}",
    )

    available_shields = [shield.identifier for shield in client.shields.list()]
    if not available_shields:
        print("No available shields. Disable safety.")
    else:
        print(f"Available shields found: {available_shields}")

    all_models = [
        model.identifier for model in client.models.list()
    ]
    if not all_models: 
        raise ValueError(
            "No models found. Check the /v1/models endpoint for your Llama Stack server."
        )

    available_models = [
        model.identifier for model in client.models.list() if model.model_type == "llm"
    ]
    if not available_models: 
        raise ValueError(
            "No available llm models found. Use a model of with 'model_type':'llm'."
        )
    
    # The e2e_loop_with_client_tools sample only supported Llama 3.2 models
    # supported_models = [x for x in available_models if "3.2" in x and "Vision" not in x]
    supported_models = [x for x in available_models if "Vision" not in x]
    if not supported_models:
        raise ValueError(
            "No supported models found. This client expects an LLM that is not a Vision model."
        )
    else:
        selected_model = supported_models[0]
        logger.debug(f"Using model: {selected_model}")

    client_tools = [
        #get_ticker_data,
        WebSearchTool(os.getenv("TAVILY_SEARCH_API_KEY")),
        #calculator,
    ]
    agent_config = AgentConfig(
        model=selected_model,
        instructions="You are a helpful assistant with access to the following function calls. Your task is to produce a list of function calls necessary to generate response to the user utterance. Use the following function calls as required.", # For Granite
        #instructions="You are a helpful assistant. Use the tools you have access to for providing relevant answers.", # For Llama
        sampling_params={
            "strategy": {"type": "top_p", "temperature": 1.0, "top_p": 0.9},
        },
        toolgroups=[
            "builtin::code_interpreter",
            "builtin::websearch",
        ],
        client_tools=[
            client_tool.get_tool_definition() for client_tool in client_tools
        ],
        tool_choice="auto",
        tool_prompt_format="json", # for Granite
        #tool_prompt_format="python_list", # For Llama
        input_shields=available_shields if available_shields else [],
        output_shields=available_shields if available_shields else [],
        enable_session_persistence=False,
    )

    agent = Agent(client, agent_config, client_tools)
    session_id = agent.create_session("test-session")
    print(f"Created session_id={session_id} for Agent({agent.agent_id})")

    user_prompts = [
        "Who was the 42nd president of the United States?",
        "What is 41+30?",
        #"Who won the Super Bowl in 2025 according to a web search?",
        "How fast can a cheetah run?",
        "How long would it take a cheetah to run across the Pont Des Artes?"
    ]
    for prompt in user_prompts:
        cprint(f"User> {prompt}", "green")
        response = agent.create_turn(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            session_id=session_id,
        )
        for log in EventLogger().log(response):
            log.print()


def main(host: str, port: int):
    asyncio.run(run_main(host, port))


if __name__ == "__main__":
    fire.Fire(main)
