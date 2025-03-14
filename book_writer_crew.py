import os
from crewai import Agent, Task, Crew, Process
from litellm import completion
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional

# Set environment variables for OpenRouter API
os.environ["OPENROUTER_API_KEY"] = "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"

# Create a custom LLM class that uses LiteLLM with OpenRouter
class LiteLLMOpenRouter(LLM):
    model_name: str = "openrouter/meta-llama/llama-3.3-70b-instruct:free"
    transforms: List[str] = []
    route: str = ""
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = completion(
            model=self.model_name,
            messages=messages,
            transforms=self.transforms,
            route=self.route
        )
        return response.choices[0].message.content
    
    @property
    def _llm_type(self) -> str:
        return "custom_litellm_openrouter"
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_name": self.model_name}

# Initialize our custom LLM
openrouter_llm = LiteLLMOpenRouter(
    model_name="openrouter/meta-llama/llama-3.3-70b-instruct:free",
    transforms=[],
    route=""
)

# Defining the Agents with OpenRouter LLM
planning_agent = Agent(
    role="Planning Agent",
    goal="Develop the book's concept, outline, characters, and world.",
    backstory="An experienced author specializing in planning and structuring novels.",
    llm=openrouter_llm,
    verbose=True
)

# Define the Writing Agent
writing_agent = Agent(
    role="Writing Agent",
    goal="Write detailed chapters based on the provided outline and character details.",
    backstory="A creative writer adept at bringing stories to life.",
    llm=openrouter_llm,
    verbose=True
)

# Define the Editing Agent
editing_agent = Agent(
    role="Editing Agent",
    goal="Edit the written chapters for clarity, coherence, and grammatical accuracy.",
    backstory="A meticulous editor with an eye for detail.",
    llm=openrouter_llm,
    verbose=True
)

# Define the Fact-Checking Agent
fact_checking_agent = Agent(
    role="Fact-Checking Agent",
    goal="Verify the accuracy of all factual information presented in the book.",
    backstory="A diligent researcher ensuring all facts are correct.",
    llm=openrouter_llm,
    verbose=True
)

# Define the Publishing Agent
publishing_agent = Agent(
    role="Publishing Agent",
    goal="Format the manuscript and prepare it for publication.",
    backstory="An expert in publishing standards and formatting.",
    llm=openrouter_llm,
    verbose=True
)

# Define the tasks for each agent
tasks = [
    Task(
        description="Develop the book's concept, outline, characters, and world.",
        expected_output="A comprehensive plan including theme, genre, outline, character profiles, and world details.",
        agent=planning_agent
    ),
    Task(
        description="Write detailed chapters based on the provided outline and character details. Each chapter should be 1000 words at least",
        expected_output="Drafts of all chapters in the book.",
        agent=writing_agent
    ),
    Task(
        description="Edit the written chapters for clarity, coherence, and grammatical accuracy.",
        expected_output="Edited versions of all chapters.",
        agent=editing_agent
    ),
    Task(
        description="Verify the accuracy of all factual information presented in the book.",
        expected_output="A report confirming the accuracy of all facts or detailing necessary corrections.",
        agent=fact_checking_agent
    ),
    Task(
        description="Format the manuscript and prepare it for publication.",
        expected_output="A finalized manuscript ready for publication.",
        agent=publishing_agent
    )
]

# Assembling all the Agents 
book_writing_crew = Crew(
    agents=[planning_agent, writing_agent, editing_agent, fact_checking_agent, publishing_agent],
    tasks=tasks,
    process=Process.sequential,
    verbose=True
)

# Execute the workflow
if __name__ == "__main__":
    result = book_writing_crew.kickoff()
    print("Final Manuscript:", result)
