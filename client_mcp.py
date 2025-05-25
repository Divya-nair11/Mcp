# lang_client.py
import asyncio
import sys
from typing import Optional
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage

class BedrockMCPClient:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.session: Optional[ClientSession] = None
        self.agent: Optional[AgentExecutor] = None

    async def connect_to_server(self, server_script_path: str):
        """Connect to MCP server and initialize agent"""
        server_params = StdioServerParameters(
            command="python",
            args=[server_script_path],
        )

        # Connect to MCP server
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(*stdio_transport)
        )
        await self.session.initialize()

        # Load tools and create agent
        tools = await load_mcp_tools(self.session)
        self._setup_bedrock_agent(tools)

    def _setup_bedrock_agent(self, tools):
        """Configure Bedrock agent with automatic tool routing"""
        llm = ChatBedrock(
            provider="anthropic",
            model_id="",
            temperature=0,
            max_tokens=3000,
            region_name='us-east-2',
            aws_access_key_id="",
            aws_secret_access_key="",
        )

        # Inside _setup_bedrock_agent():
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant skilled in math. Use tools when available. If a tool is not relevant, answer directly using your own knowledge."),

            ("user", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])

        
        agent = create_tool_calling_agent(llm, tools, prompt)
        self.agent = AgentExecutor(agent=agent, tools=tools, verbose=True)

    async def process_query(self, query: str) -> str:
        """Process natural language queries with tool fallback"""
        if not self.agent:
            raise RuntimeError("Agent not initialized")

        try:
            print("[MCP] AgentExecutor invoked.")
            result = await self.agent.ainvoke({"input": query})
            output = result.get("output")
            if isinstance(output, str):
                return "[MCP Response] " + output
            elif isinstance(output, list):
                return "[MCP Response] " + " ".join(str(o) for o in output)
            else:
                return "[LLM Response] " + str(output)
        except Exception as e:
            print(f"[LLM Fallback] Exception occurred: {e}")
            llm = ChatBedrock(
                provider="anthropic",
                model_id="",
                temperature=0,
                max_tokens=3000,
                region_name='us-east-2',
                aws_access_key_id="",
                aws_secret_access_key="",
            )
            result = await llm.ainvoke([HumanMessage(content=query)])
            return "[LLM Fallback] " + result.content



    async def chat_loop(self):
        """Interactive chat interface"""
        print("Math Assistant (type 'quit' to exit)")
        while True:
            try:
                query = input("\nQuestion: ").strip()
                if query.lower() == 'quit':
                    break
                response = await self.process_query(query)
                print("\n" + response)
            except KeyboardInterrupt:
                break

    async def cleanup(self):
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python lang_client.py <path_to_server_script>")
        sys.exit(1)

    client = BedrockMCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())