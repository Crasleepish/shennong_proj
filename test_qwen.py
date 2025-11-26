from app.ai.qwen_client import QwenClient

def main():
    client = QwenClient(
        model="qwen-plus",
        default_params={
            "temperature": 0.1,
            "top_p": 1.0,
            "max_tokens": 1024,
        },
    )

    # 1）普通调用
    answer = client.chat([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "帮我生成一个两段式邮件模板，用来向客户催要发票。"},
    ])
    print(answer)

    # 2）流式调用
    print("\n=== 流式输出 ===")
    for chunk in client.stream_chat([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "帮我生成一个三段式邮件模板，用来向客户催要发票。"},
    ]):
        print(chunk, end="", flush=True)

if __name__ == "__main__":
    main()