from helper import add_ai_message, add_user_message, chat
import json
import ast
import re


def grade_by_model(testcase, output):
    eval_prompt = f"""
                You are an expert AWS code reviewer. Your task is to evaluate the following AI-generated solution.

                Original Task:
                <task>
                {testcase["task"]}
                </task>

                Solution to Evaluate:
                <solution>
                {output}
                </solution>

                Criteria you should use to evaluate the solution:
                <criteria>
                {testcase["solution_criteria"]}
                </criteria>

                Output Format
                Provide your evaluation as a structured JSON object with the following fields, in this specific order:
                - "strengths": An array of 1-3 key strengths
                - "weaknesses": An array of 1-3 key areas for improvement
                - "reasoning": A concise explanation of your overall assessment
                - "score": A number between 1-10

                Respond with JSON. Keep your response concise and direct.
                Example response shape:
                {{
                    "strengths": string[],
                    "weaknesses": string[],
                    "reasoning": string,
                    "score": number
                }}
                """
    
    messages = []
    add_user_message(messages, eval_prompt)
    add_ai_message(messages, "```json")
    eval_text = chat(messages, stop_sequences=["```"])

    return json.loads(eval_text)


# Code based grading below to bottom of file #
def validate_json(text):
    try:
        json.loads(text.strip())
        return 10
    except json.JSONDecodeError:
        return 0


def validate_python(text):
    try:
        ast.parse(text.strip())
        return 10
    except SyntaxError:
        return 0


def validate_regex(text):
    try:
        re.compile(text.strip())
        return 10
    except re.error:
        return 0


def grade_syntax(response, test_case):
    format = test_case["format"]
    if format == "json":
        return validate_json(response)
    elif format == "python":
        return validate_python(response)
    else:
        return validate_regex(response)
