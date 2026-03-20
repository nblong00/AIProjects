from helper import add_ai_message, add_user_message, chat
from grader import grade_by_model, grade_syntax
from statistics import mean


def generate_dataset_ai():
    prompt = """
            Generate a evaluation dataset for a prompt evaluation. The dataset will be used to evaluate prompts
            that generate Python, JSON, or Regex specifically for AWS-related tasks. Generate an array of JSON objects,
            each representing task that requires Python, JSON, or a Regex to complete.

            Example output:
            ```json
            [
                {
                    "task": "Description of task",
                    "format": "json", or "python" or "regex",
                    "solution_criteria": "Key criteria for evaluating the solution"
                },
                ...additional
            ]
            ```

            * Focus on tasks that can be solved by writing a single Python function, a single JSON object, or a regular expression.
            * Focus on tasks that do not require writing much code

            Please generate 3 objects.
            """
    
    return prompt


def run_prompt(testcase):
    prompt = f"""
            Please solve the following task: 
            {testcase["task"]}
            * Respond only with Python, JSON, or a plan Regex. 
            * Do not add any comments or commentary or explanation.
            """
    messages = []
    add_user_message(messages, prompt)
    add_ai_message(messages, "```code")
    output = chat(messages, stop_sequences=["```"])

    return output


def run_test_case(test_case):
    output = run_prompt(test_case)

    model_grade = grade_by_model(test_case, output)
    model_score = model_grade["score"]
    reasoning = model_grade["reasoning"]

    syntax_score = grade_syntax(output, test_case)
    score = (model_score + syntax_score) / 2

    return {
        "output": output,
        "test_case": test_case,
        "score": score,
        "reasoning": reasoning
    }


def run_eval(dataset):
    results = []
    for test_case in dataset:
        result = run_test_case(test_case)
        results.append(result)

    average_score = mean([result["score"] for result in results])
    print(average_score)

    return results
