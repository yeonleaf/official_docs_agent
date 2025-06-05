import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
import logging

logger = logging.getLogger(__name__)

load_dotenv()

llm = AzureChatOpenAI(
    openai_api_key=os.getenv("AOAI_API_KEY"),
    azure_endpoint=os.getenv("AOAI_ENDPOINT"),
    azure_deployment=os.getenv("AOAI_DEPLOY_GPT4O"),
    api_version=os.getenv("AOAI_API_VERSION"),
    temperature=0.7,
)
embeddings = AzureOpenAIEmbeddings(model=os.getenv('AOAI_DEPLOY_EMBED_3_LARGE'),
    openai_api_version="2024-02-01",
    api_key= os.getenv("AOAI_API_KEY"),  
    azure_endpoint=os.getenv("AOAI_ENDPOINT")
)

def call_llm_with_backoff(messages, max_retries=5):
    for retry in range(max_retries):
        try:
            return llm.invoke(messages)
        except openai.error.RateLimitError:
            wait = 2 ** retry  # 1, 2, 4, 8, 16초
            logger.info(f"[RateLimit] {retry+1}회차 재시도: {wait}초 대기")
            time.sleep(wait)
    raise RuntimeError("LLM 호출 재시도 실패")