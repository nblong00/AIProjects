from promptEvalWorkflow import generate_dataset_ai, run_eval
from helper import (add_ai_message,
                    add_user_message,
                    export_results,
                    chat)
import json


def main():
    messages = []
    add_user_message(messages, generate_dataset_ai())
    add_ai_message(messages, "```json")
    answer = json.loads(chat(messages, stop_sequences=["```"]))
    results = run_eval(answer)
    export_results(results)


main()
