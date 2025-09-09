import os
from crewai import Agent
from langchain_cohere import ChatCohere
from langchain_openai import OpenAI
from pydantic import SecretStr


llm = OpenAI(model="gpt-4o-mini", temperature=0)
llm_cohere = ChatCohere(
    cohere_api_key=SecretStr(os.getenv("COHERE_API_KEY", "")), temperature=0
)

developer_agent = Agent(
    role="Senior Python Backend Developer",
    goal="You are a senior python backend developer",
    backstory="You are a senior python backend developer with 10 years of experience",
    llm=llm,
)


requirements_agent = Agent(
    role="Requirements Analyst",
    goal="Understand user intent and create detailed requirements",
    backstory="You specialize in understanding what users really want and turning that into dev-ready specs.",
    llm=llm_cohere,
)

qa_agent = Agent(
    role="QA Tester",
    goal="Create unit tests and verify code correctness",
    backstory="You test software for correctness, edge cases, and reliability.",
    llm=llm,
)
