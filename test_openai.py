
import openai
from openai import OpenAI

client = OpenAI(
organization='YOUR_ORG_ID',
api_key='empty',
base_url='http://172.16.13.199:8090/v1'
)
model = "text-embedding-ada-002"

result = client.embeddings.create(input=['你好'],model=model).data[0].embedding
print(result)