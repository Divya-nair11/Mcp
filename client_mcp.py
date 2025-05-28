import asyncio
import sys
import re
from typing import Optional
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent

class BedrockMCPClient:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.session: Optional[ClientSession] = None
        self.agent: Optional[AgentExecutor] = None
        self.llm = None
        self.tools = []

    async def connect_to_server(self, server_script_path: str):
        server_params = StdioServerParameters(
            command="python",
            args=[server_script_path],
        )
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.session = await self.exit_stack.enter_async_context(ClientSession(*stdio_transport))
        await self.session.initialize()

        self.tools = await load_mcp_tools(self.session)
        self._setup_bedrock_agent(self.tools)

    def _setup_bedrock_agent(self, tools):
        self.llm = ChatBedrock(
            provider="anthropic",
            model_id="",
            temperature=0,
            max_tokens=3000,
            region_name='us-east-2',
            aws_access_key_id="",
            aws_secret_access_key="",
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful math assistant. The query includes relevant math context.
If the context answers the query, respond directly. Otherwise use appropriate math tools.
For non-math queries, answer directly using your knowledge."""),
            ("user", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])

        agent = create_tool_calling_agent(self.llm, tools, prompt)
        self.agent = AgentExecutor(agent=agent, tools=tools, verbose=True)

    def _extract_numbers(self, query: str):
        parts = re.split(r'\D+', query)  
        return [int(p) for p in parts if p.isdigit()]


    async def _get_relevant_addition(self, query: str):
        """Always perform an add call, using extracted numbers if possible, else defaults"""
        numbers = self._extract_numbers(query)
        if len(numbers) >= 2:
            a, b = numbers[:2]
        else:
            a, b = 1, 2  # default values for testing
        result = await self.session.call_tool("add", {"a": a, "b": b})
        return f"Relevant addition: {a} + {b} = {result}", a, b


    async def process_query(self, query: str, recursion_depth: int = 0) -> str:
        if not self.agent or not self.session:
            raise RuntimeError("Agent not initialized")
        
        if recursion_depth > 3:
            return "Error: Maximum tool calls exceeded"

        # Add relevant math context
        math_context, a, b = await self._get_relevant_addition(query)
        extended_query = (
            f"User Question: {query}\n\n"
            f"[Note: The following is irrelevant math context (used for tracking only): {math_context}]\n"
            f"Please answer the user's original question as the main priority."
        )

        try:
            #  Try with agent first
            result = await self.agent.ainvoke({"input": extended_query})
            output = result.get("output", "")

            # Check if tool should be called
            if isinstance(output, str):
                tool_names = [t.name for t in self.tools if t.name != "add"]

                found_tool = None
                for tool_name in tool_names:
                    if tool_name.lower() in output.lower():
                        found_tool = tool_name
                        break

                if found_tool:
                    
                    tool_result = await self.session.call_tool(found_tool, {"a": a, "b": b})
                    new_query = f"{query}\n{found_tool}({a}, {b}) = {tool_result}"
                    return await self.process_query(new_query, recursion_depth + 1)
            # Clean up the output format
            if isinstance(output, list):
                output = " ".join(str(item) for item in output)
            return f"[Final Answer] {output}"

        except Exception as e:
            # Fallback to direct LLM response
            print(f"Tool error, falling back to LLM: {str(e)}")
            llm_response = await self.llm.ainvoke(extended_query)
            return f"[LLM Response] {llm_response.content}"

    async def chat_loop(self):
        print("Assistant (type 'quit' to exit)")
        while True:
            try:
                query = input("\nQuestion: ").strip()
                if query.lower() == 'quit':
                    break
                response = await self.process_query(query)
                print(response)
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
    
