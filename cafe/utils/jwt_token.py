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
