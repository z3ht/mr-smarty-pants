from openai import OpenAI, AsyncOpenAI
from msp.settings import OPENAI_API_KEY, LAMBDA_API_KEY


def get_client(model_name: str):
    is_openai_model = model_name.startswith(("gpt-", "gpt4", "gpt3", "text-davinci", "code-davinci"))
    is_vision_model = "vision" in model_name or "llava" in model_name or "vlm" in model_name

    ClientClass = OpenAI if is_vision_model else AsyncOpenAI

    if is_openai_model:
        return ClientClass(api_key=OPENAI_API_KEY)
    else:
        return ClientClass(
            api_key=LAMBDA_API_KEY,
            base_url="https://api.lambda.ai/v1"
        )
