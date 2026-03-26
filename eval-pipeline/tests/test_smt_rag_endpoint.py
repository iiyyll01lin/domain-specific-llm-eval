import requests


def call_smt_assistant(url, param):
    headers = {
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(url, json=param, headers=headers)
        return response.json()["message"][0]['content']
    except requests.exceptions as e:
        print(f"[ERROR] Request failed: {e}")
        return "[ERROR] Request failed"


if __name__ == '__main__':
    question = "遇到錯誤代碼960D0000怎麼辦？"
    parameter = {
        "content": question,
        "session_id": "",
        "command": "",
        "context": {},
        "app": "smt_assistant_chat",
        "user": "",
        "site": "tao",
        "language": "en"
    }
    smt_assistant_url = "http://10.3.30.13:8855/app/smt_assistant_chat"
    result = call_smt_assistant(smt_assistant_url, parameter)
    print(result)
