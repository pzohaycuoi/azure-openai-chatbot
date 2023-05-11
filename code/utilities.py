import tiktoken


def count_token(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    count_token = len(encoding.encode(text))
    
    return count_token
