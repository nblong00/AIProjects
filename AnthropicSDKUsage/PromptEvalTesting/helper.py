from anthropic import Anthropic
from dotenv import load_dotenv
import json

load_dotenv()
client = Anthropic()
model = 'claude-sonnet-4-5'


def add_user_message(messages, text):
    user_message = {"role": "user", "content": text}
    messages.append(user_message)


def add_ai_message(messages, text):
    ai_message = {"role": "assistant", "content": text}
    messages.append(ai_message)


def chat(messages, system=None, temperature=1.0, stop_sequences=[]):
    params = {
        "model": model,
        "max_tokens": 1000,
        "messages": messages,
        "temperature": temperature,
        "stop_sequences": stop_sequences,
    }

    if system:
        params["system"] = system

    message = client.messages.create(**params)

    return message.content[0].text


def export_results(answer):
    with open("dataset.json", "w") as f:
        json.dump(answer, f, indent=2)
