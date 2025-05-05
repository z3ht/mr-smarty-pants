from pprint import pprint
from openai import OpenAI

from msp.settings import LAMBDA_API_KEY, OPENAI_API_KEY

lambda_client = OpenAI(
    api_key=LAMBDA_API_KEY,
    base_url="https://api.lambda.ai/v1"
)

# cheaper
# deepseek is very good!
print("LAMBDA MODELS:")
pprint(lambda_client.models.list().data)

# industry standard
openai_client = OpenAI(
    api_key=OPENAI_API_KEY
)
print("OPEN AI MODELS:")
pprint(openai_client.models.list().data)

