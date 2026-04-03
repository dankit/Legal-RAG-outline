from langchain_google_vertexai import ChatVertexAI
from ..search.search_agent import SearchAgent
from langchain_core.messages import SystemMessage, HumanMessage
from typing import List, Dict, Any
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from ..tools.chat_tools import SimpleSearchTool, SequentialSearchTool
from ..prompts.chat_agent_prompts import (
    build_chat_agent_system_prompt,
    get_conversation_history_summarization_prompt,
    get_sequential_question_breakdown_prompt,
    get_sequential_results_synthesis_prompt
)

logger = logging.getLogger(__name__)


class ChatAgent:
    """Conversational agent for legal document search and Q&A.
    
    Supports two search modes:
    - SimpleSearchTool: for straightforward questions answered with a single search
    - SequentialSearchTool: for multi-part questions where later steps depend on earlier results
    """
    
    def __init__(self, conversation_history_limit: int = 10):
        self.llm = ChatVertexAI(model="gemini-2.5-flash", temperature=0)
        self.conversation_history: List[str] = []
        self.conversation_history_summaries: List[str] = []
        self.conversation_history_limit = conversation_history_limit
        self.sequential_steps: List[Dict[str, Any]] = []
        self.max_sequential_steps = 3
        self.total_token_usage = 0

    def _summarize_conversation_history(self) -> None:
        """Compress conversation history to save tokens using rolling window approach."""
        try:
            history_to_summarize = self.conversation_history[:self.conversation_history_limit // 2]
            prompt = get_conversation_history_summarization_prompt(history_to_summarize)
            
            summarization = self.llm.invoke([
                SystemMessage(content="You are a helpful assistant that summarizes conversation history."),
                HumanMessage(content=prompt)
            ])
            self.conversation_history_summaries.append(summarization.content)
            self.conversation_history = self.conversation_history[self.conversation_history_limit // 2:]
            
        except Exception as e:
            logger.error(f"Failed to summarize conversation history: {e}")
    
    def chat(self, user_input: str) -> str:
        """Process user input and return response, handling tool calls as needed."""
        try:
            start_time = time.time()
            system_prompt = build_chat_agent_system_prompt(self.conversation_history)
            llm_response = self.llm.invoke([
                SystemMessage(content=system_prompt), 
                HumanMessage(content=user_input)
            ], tools=[SimpleSearchTool, SequentialSearchTool])
            self.total_token_usage += llm_response.usage_metadata['total_tokens']
            answer = self._process_llm_response(llm_response)
            logger.info(f"Chat response time: {time.time() - start_time:.2f}s, tokens: {self.total_token_usage}")
            self._update_conversation_history(f"User input: {user_input}\n Agent response: {answer}\n")
            self.total_token_usage = 0
            return answer

        except Exception as e:
            logger.error(f"Error in chat processing: {e}")
            return f"I encountered an error processing your request: {str(e)}"

    def _process_llm_response(self, llm_response) -> str:
        answer = ""
        if not llm_response.tool_calls:
            return llm_response.content
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for tool_call in llm_response.tool_calls:
                logger.info(f"Chat agent tool call: {tool_call}")
                if tool_call['name'] == "SimpleSearchTool":
                    futures.append(executor.submit(self._handle_simple_search, tool_call))
                elif tool_call['name'] == "SequentialSearchTool":
                    futures.append(executor.submit(self._handle_sequential_search, tool_call))
            for future in futures:
                answer += future.result() + "\n"
        return answer
    
    def _handle_simple_search(self, tool_call: Dict[str, Any]) -> str:
        query = tool_call['args'].get('query', '')
        context = tool_call['args'].get('context', '')
        search_agent = SearchAgent()
        answer = search_agent.run(query, context)
        self.total_token_usage += search_agent._calculate_total_tokens()
        return answer
    
    def _handle_sequential_search(self, tool_call: Dict[str, Any]) -> str:
        """Handle sequential search tool calls."""
        question = tool_call['args'].get('question', '')
        
        if not question:
            logger.warning("Empty question in SequentialSearchTool call")
            return "I received an empty question for sequential processing. Please rephrase your question."
                
        try:
            return self._execute_sequential_search(question)
        except Exception as e:
            logger.error(f"Sequential search failed: {e}")
            return f"I encountered an error while processing your sequential question: {str(e)}"
    
    def _execute_sequential_search(self, question: str) -> str:
        """Execute sequential search by breaking down the question into steps."""
        self.sequential_steps = []
        
        steps = self._break_down_question(question)
        if not steps:
            return "I couldn't break down your question into logical steps. Please try rephrasing your question."
        
        logger.info(f"Broken down question into {len(steps)} steps: {steps}")
        
        step_results = []
        context = ""
        search_agent = SearchAgent()
        for i, step in enumerate(steps, 1):
            if i > self.max_sequential_steps:
                logger.warning(f"Maximum sequential steps ({self.max_sequential_steps}) reached")
                break
                
            logger.info(f"Executing step {i}: {step}")
            try:
                if step.strip() == "":
                    continue
                step_result = search_agent.run(step, context)
                step_results.append({
                    'step_number': i,
                    'query': step,
                    'result': step_result,
                    'context': context
                })
                context += f"Step {i}: {step}\nResult: {step_result}\n\n"
                
            except Exception as e:
                logger.error(f"Step {i} failed: {e}")
                step_results.append({
                    'step_number': i,
                    'query': step,
                    'result': f"Error in step {i}: {str(e)}",
                    'context': context
                })

        self.total_token_usage += search_agent._calculate_total_tokens()
        return self._synthesize_sequential_results(question, step_results)
    
    def _break_down_question(self, question: str) -> List[str]:
        prompt = get_sequential_question_breakdown_prompt()
        try:
            response = self.llm.invoke([SystemMessage(content=prompt), HumanMessage(content=f"Query: {question}")])
            steps = [step.strip() for step in response.content.split('\n') if step.strip()]
            self.total_token_usage += response.usage_metadata['total_tokens']
            return steps[:self.max_sequential_steps]
        except Exception as e:
            logger.error(f"Failed to break down question: {e}")
            return []
    
    def _synthesize_sequential_results(self, original_question: str, step_results: List[Dict[str, Any]]) -> str:
        """Synthesize results from sequential steps into a coherent answer."""
        if not step_results:
            return "I couldn't find relevant information to answer your question."
        
        synthesis_context = f"""Original Question: {original_question}\nSequential Search Results:\n"""
        for step in step_results:
            synthesis_context += f"Step {step['step_number']}: {step['query']}\nResult: {step['result']}\n\n"
        
        synthesis_prompt = get_sequential_results_synthesis_prompt(original_question, synthesis_context)
        
        try:
            response = self.llm.invoke([
                SystemMessage(content="You are a helpful assistant that synthesizes search results into comprehensive answers. Always cite either the document with page number, or FULL web source url in the answer."),
                HumanMessage(content=synthesis_prompt)
            ])
            self.total_token_usage += response.usage_metadata['total_tokens']
            return response.content
        except Exception as e:
            logger.error(f"Failed to synthesize results: {e}")
            return "\n\n".join([step['result'] for step in step_results])
    
    def _update_conversation_history(self, context: str) -> None:
        """Update conversation history and compress if needed."""
        self.conversation_history.append(context)
        if len(self.conversation_history) >= self.conversation_history_limit:
            self._summarize_conversation_history()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    agent = ChatAgent()
    print("Chat Agent initialized. Type 'quit' to exit.")
    while True:
        try:
            user_input = input("User: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            if user_input:
                response = agent.chat(user_input)
                print(f"Agent: {response}\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
