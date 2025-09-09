from dotenv import load_dotenv
from crewai import Task, Crew
from fastapi import FastAPI, Body

from agents import requirements_agent, developer_agent, qa_agent

load_dotenv()


app = FastAPI()


@app.post("/generate-code")
async def generate_code(user_request: str = Body(..., embed=True)):
    task1 = Task(
        description=f"Turn this user request into a complete software specification: '{user_request}'",
        agent=requirements_agent,
        expected_output="A complete software specification",
    )

    task2 = Task(
        description="Write working Python code based on the specification",
        agent=developer_agent,
        expected_output="A complete software specification",
        context=[task1],
    )

    task3 = Task(
        description="Write test cases for the code to ensure it's functioning properly",
        agent=qa_agent,
        expected_output="A complete software specification",
        context=[task2],
    )

    crew = Crew(
        agents=[requirements_agent, developer_agent, qa_agent],
        tasks=[task1, task2, task3],
        verbose=True,
    )

    result = crew.kickoff()

    return {
        "final_output": result,
    }
