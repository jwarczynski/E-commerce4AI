import requests


def call_cortex_tool(jwt_token: str, snowflake_host: str, prompt: str) -> dict:
    url = f"https://{snowflake_host}/api/v2/cortex/inference:complete"

    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    data = {
        "model": "claude-3-5-sonnet",
        "messages": [{"role": "user", "content": prompt}],
        "tools": [
            {
                "tool_spec": {
                    "type": "generic",
                    "name": "get_weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ],
        "max_tokens": 4096,
        "top_p": 1,
        "stream": False
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()


def parse_tool_call(response_json: dict) -> dict:
    try:
        message = response_json["choices"][0]["message"]
        content_list = message.get("content_list", [])

        tool_call = next((item["tool_use"] for item in content_list if item["type"] == "tool_use"), None)
        text_content = " ".join(item["text"] for item in content_list if item["type"] == "text")

        if tool_call:
            return {
                "tool_name": tool_call["name"],
                "arguments": tool_call["input"],
                "text_output": text_content
            }
    except (KeyError, IndexError, TypeError):
        pass

    return {"tool_name": None, "arguments": {}, "text_output": ""}
