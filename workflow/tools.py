from langchain.tools import tool
from duckduckgo_search import DDGS
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
import os
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import requests
from urllib.parse import urlparse, urljoin
import asyncio
import aiohttp
from bs4 import BeautifulSoup, Comment
from typing import List
from pydantic import BaseModel
import streamlit as st
import requests
import re
import sys
from db.repository import enqueue_request, read_url, insert_url
import time
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from config.llm_client import llm, embeddings
from config.chroma_client import client, get_chroma_collection
import tiktoken
from pathlib import Path
from duckduckgo_search.exceptions import (
    ConversationLimitException,
    DuckDuckGoSearchException,
    RatelimitException,
    TimeoutException,
)
import random
import logging
logger = logging.getLogger(__name__)

ddgs = DDGS()

@tool
def get_general_keyword_docs(keyword):
    """
    주어진 키워드(Computer Science와 관련된)에 대해 공식적인 문서의 URL와 HTML 본문을 반환합니다.
    Args:
        keyword (str): 임베딩 대상인 일반 Computer Science 키워드
    """
    safe_urls = [
        "https://developer.mozilla.org/en-US/docs/Glossary/",
        "https://en.wikipedia.org/wiki/"
    ]
    for safe_url in safe_urls:
        search_keyword = "_".join(keyword.strip().lower().split(" "))
        search_url = urljoin(safe_url, search_keyword)

        response = requests.get(search_url)
        if "Page not found" in response.text or "Wikipedia does not have an article with this exact name" in response.text:
            continue
        
        refined_text = clean_html_for_text(response.text)
        return [(search_url, refined_text)]
    return None

@tool
def embed_general_docs(keyword: str, responses: List):
    """
    주어진 키워드와 관련된 HTML 응답 리스트를 정제 및 요약한 후,
    벡터 DB(Chroma)에 임베딩하여 저장하는 함수입니다.

    각 HTML 응답에서 본문 텍스트를 추출(cleaning)하고 요약한 뒤,
    요약된 텍스트의 길이가 일정 기준(800자 이상)을 넘을 경우
    해당 내용을 Chroma에 저장합니다.

    매 처리마다 요약된 텍스트를 stdout에 출력하여 진행 상황을 확인할 수 있습니다.

    Args:
        keyword (str): 임베딩 대상인 일반 Computer Science 키워드
        responses (List): (url, html) 형태의 튜플 리스트
    """
    for url, html in responses:
        if html:
            text = clean_html_for_text(html)
            if len(text) > 800:
                persist_chroma("general", keyword, text, url)

def search_ddgs(query: str,
                max_results: int = 10,
                max_retry: int = 5,
                base_delay: float = 1.5,
                jitter: float = 0.3) -> list[dict] | None:
    """
    DDG 검색 → 429 발생 시 지수 백오프 + Jitter 재시도.

    Args:
        query        : 검색어
        max_results  : 최대 결과 수
        max_retry    : 재시도 한계
        base_delay   : 첫 대기 시간(초)
        jitter       : 무작위 지터 범위(±)

    Returns:
        list[dict] | None : 성공 시 결과, 실패 시 None
    """
    for attempt in range(max_retry):
        try:
            # ① 호출
            return ddgs.text(
                query,
                region="en",
                safesearch="moderate",
                max_results=max_results,
            )
        except (RatelimitException, TimeoutException) as e:
            wait = base_delay * (2 ** attempt)
            wait += random.uniform(-jitter, jitter)
            logger.warning(
                f"[DDG] 429/Timeout 발생 - {wait:.1f}s 후 재시도 "
                f"(시도 {attempt+1}/{max_retry})"
            )
            time.sleep(max(wait, 0))
        except Exception:
            raise
    logger.error("[DDG] 최대 재시도 초과")
    return None

def it_seems_official(url):
    """
    주어진 URL이 공식 문서로 보이는지를 판단합니다.

    Args:
        url (str): 검증할 웹 페이지의 URL.

    Returns:
        bool: URL이 공식 문서 경로로 보이면 True, 그렇지 않으면 False.

    Logic:
        - URL의 경로에 'docs', 'reference', 'guide' 등 문서 관련 키워드가 포함되어야 합니다.
        - 동시에 'github', 'blog', 'tutorial', 'medium' 등의 비공식 출처 키워드가 URL 전체에 포함되어 있으면 False를 반환합니다.
    """
    DOC_PATH_KEYWORDS = ["docs", "reference", "guide", "readme", "learn"]
    BAD_KEYWORDS = ["github", "blog", "tutorial", "wiki", "baeldung", "medium", "stackoverflow", "dev.to", "api"]  
    path = urlparse(url).path.lower()
    return (any(k in path for k in DOC_PATH_KEYWORDS)
            and not any(b in url for b in BAD_KEYWORDS))

@tool
def get_official_url(framework: str) -> str:
    """
    주어진 framework 이름에 대해 공식적인 문서의 URL을 웹에서 검색한 후 올바른 공식 문서 URL인지 검증한 후 반환합니다.
    """

    collection_nm = norm(framework)
    existing_url = read_url(collection_nm)
    if existing_url:
        return existing_url

    query = f"{framework} current official documentation"
    result = None
    try:
        results = search_ddgs(query)
    except RatelimitException as e:
        return None
    official_urls = []
    for result in results:
        url = result.get("href", "")
        if it_seems_official(url):
            official_urls.append(url)
    official_urls = sorted(official_urls, key=len)
    if len(official_urls) == 0:
        return None
    official_url = official_urls[0]
    insert_url(collection_nm, official_url)
    return official_url

def clean_html_for_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # script, style, noscript, iframe 제거
    for tag in soup(["script", "style", "noscript", "iframe"]):
        tag.decompose()

    # HTML 주석 제거
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # 텍스트 추출
    text = soup.get_text(separator=" ")

    # 공백 정리: 연속된 공백 → 하나로, 양쪽 공백 제거
    text = re.sub(r"\s+", " ", text).strip()

    return text

async def safe_fetch(session, semaphore, url, clean=False):
    async with semaphore:
        return await fetch_with_retry(session, url, clean)

async def fetch_with_retry(session, url, clean=False, retries=3, delay=1):
    for attempt in range(retries):
        try:
            async with session.get(url, ssl=False) as response:
                response_text = await response.text()
                if clean:
                    return url, clean_html_for_text(response_text)
                return url, response_text
        except Exception as e:
            if attempt < retries - 1:
                await asyncio.sleep(delay)
            else:
                logger.error(f"[에러] {url} - fetch 실패: {type(e)} {e}")
                return None

def is_valid_doc_url(href: str) -> bool:
    if not href:
        return False
    if href.startswith("javascript:") or href.startswith("#"):
        return False
    if any(ext in href for ext in [".css", ".js", ".png", ".jpg", ".svg", ".ico", ".pdf", ".zip", ".json"]):
        return False
    # 핵심 조건 수정: ".html"로 끝나거나 "/"로 끝나거나, 또는 끝에 확장자가 없는 경로도 허용
    parsed = urlparse(href)
    path = parsed.path
    if path.endswith("/") or path.endswith(".html"):
        return True
    # 확장자 없는 단순 경로
    if "." not in path.split("/")[-1]:  # 마지막 경로에 점이 없으면 확장자 없는 문서형 경로로 간주
        return True
    return False

async def _crawl_docs(seed_url: str, semaphore):
    async with aiohttp.ClientSession() as session:
        _, seed_html = await safe_fetch(session, semaphore, seed_url, clean=False)
        if not seed_html:
            return []
        soup = BeautifulSoup(seed_html, "html.parser")
        links = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if is_valid_doc_url(href):
                url = urljoin(seed_url, href).split("#")[0]
                links.add(url)
        tasks = [safe_fetch(session, semaphore, link, clean=True) for link in links]
        return await asyncio.gather(*tasks)

# 임베딩
def persist_chroma(collection_name, search_topic, text, source_url):
    collection_nm = norm(collection_name)
    db = get_chroma_collection(collection_nm)
    docs_with_source = db.get(where={"source": {"$eq": source_url}})
    if docs_with_source["documents"]:
        return
    chunks = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100).split_text(text)
    chunks = [c for c in chunks if len(c) > 100][:20]
    docs = [Document(page_content=c, metadata={"source": source_url, "fw": search_topic}) for c in chunks]
    db.add_documents(docs)

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def summarizes_text(
    text: str) -> str:
    """
    큰 문서는 계층형 요약, 중간 크기는 원문 chunk 임베딩,
    작은 문서는 그대로 반환. keywords 주어지면 그 문맥을 앞에 붙임.
    """
    token_count = num_tokens_from_string(text, "o200k_base")

    # 대용량 문서: 계층형 요약
    if token_count > 96000:
        splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
        chunks = splitter.split_text(text)
        partial_summaries = []
        for chunk in chunks:
            summary = llm.invoke(
                f"다음 내용을 350자 이내 한국어로 구체적으로 요약:\n\n{chunk}"
            ).content.strip()
            partial_summaries.append(summary)
        final = llm.invoke(
            "다음 요약들을 1500자 이내 한국어로 통합:\n\n" + "\n\n".join(partial_summaries)
        ).content.strip()
        return final

    # 중간 크기 문서: 요약하지 않고 그대로(=> 세부 용어 보존)
    if token_count > 12000:
        return text

    # 소형 문서: 1-shot 요약
    final = llm.invoke(
        f"다음 텍스트를 500자 이내 한국어로 세부 사항을 포함하여 요약:\n\n{text}"
    ).content.strip()
    return final

def embed_docs(framework: str, responses: List):
    for item in responses:
        if not item:
            continue
        url, html = item
        if html:
            text = clean_html_for_text(html)
            if len(text) > 300:
                persist_chroma(framework, framework, text, url)

class InvalidFrameworkNameException(Exception):
    pass

def norm(name: str) -> str:
    res = re.sub(r'[^a-z0-9]', '', name.lower())
    if not res:
        raise InvalidFrameworkNameException()
    return res

class AmbiguousFrameworkNameException(Exception):
    pass

def collection_exists(name: str) -> bool:
    EXCLUDED_AMBIGUOUS_PAIRS = {
        # Java / Spring
        ("spring", "springboot"),
        ("spring", "springcloud"),
        ("spring", "springsecurity"),
        ("spring", "springdata"),
        ("spring", "springwebflux"),
        ("spring", "springbatch"),

        # Hibernate
        ("hibernate", "hibernatevalidator"),
        ("validator", "hibernatevalidator"),

        # React / Vue / Node
        ("react", "reactnative"),
        ("react", "reactrouter"),
        ("react", "reactredux"),
        ("vue", "vuex"),
        ("vue", "vuerouter"),
        ("next", "nextauth"),
        ("nextjs", "nextauth"),
        ("express", "expressvalidator"),
        ("express", "expresssession"),
        ("express", "expressratelimit"),
        ("redux", "reduxtoolkit"),

        # Python
        ("flask", "flasklogin"),
        ("flask", "flaskmail"),
        ("flask", "flaskjwtextended"),
        ("django", "djangorestframework"),
        ("django", "djangochannels"),
        ("django", "djangoadmin"),

        # Cloud / DevOps
        ("terraform", "terraformcloud"),
        ("kubernetes", "kubernetesdashboard"),
        ("helm", "helmfile"),

        # Node.js utils
        ("axios", "axiosmockadapter"),
        ("eslint", "eslintpluginreact"),
        ("jest", "jestextended"),
    }

    short_count = 0
    short_existing = []
    long_count = 0
    long_existing = []
    n = norm(name)
    
    for col in client.list_collections():
        m = norm(col)
        if n == m:
            return True
        if n in m:
            # n (입력값) ⊂ m (기존): 입력이 짧은 이름 → 기존이 긴 이름
            long_count += 1
            long_existing.append(m)
        elif m in n and (m, n) not in EXCLUDED_AMBIGUOUS_PAIRS:
            # m (기존값) ⊂ n (입력): 입력이 긴 이름 → 기존이 짧은 이름
            short_count += 1
            short_existing.append(m)

    if short_count >= 1:
        short_existing_str = ", ".join(short_existing)
        raise AmbiguousFrameworkNameException(
            f"'{name}'은(는) 이미 존재하는 짧은 이름 '{short_existing_str}'과 모호하게 일치합니다. "
            f"'{short_existing_str}' 중 하나를 사용하거나, 완전히 다른 프레임워크명을 사용해 주세요."
        )
    
    if long_count >= 1:
        long_existing_str = ", ".join(long_existing)
        raise AmbiguousFrameworkNameException(
            f"'{name}'은(는) 이미 존재하는 긴 이름 '{long_existing_str}'과 모호하게 일치합니다. "
            f"'{long_existing_str}' 중 하나를 사용하거나, 완전히 다른 프레임워크명을 사용해 주세요."
        )
    
    return False
    
@tool
def verify_framework(framework: str, seed_url: str, question_id: str):
    """
    지정한 프레임워크 이름에 해당하는 컬렉션이 Chroma 벡터 DB에 존재하는지 확인하고,
    존재하지 않을 경우 문서 수집 작업을 큐에 등록합니다.

    이 함수는 RAG 기반 문서 검색 전에 해당 프레임워크의 벡터 컬렉션이 준비되어 있는지 
    사전 확인하기 위해 사용됩니다. 컬렉션이 존재하지 않으면 사용자 질문에 즉시 응답하지 않고,
    백그라운드 워커를 통해 문서 수집 및 임베딩이 이루어지도록 큐에 등록합니다.

    Args:
        framework (str): 확인할 프레임워크 이름 (벡터 DB의 collection name). (required, not None)
        seed_url (str): 공식 문서의 시작 URL. 컬렉션이 없을 경우 수집 작업에 사용됨. (required, not None)
        question_id (str): 질문 ID (required, not None)

    Returns:
        bool: 
            - True: 해당 프레임워크의 벡터 컬렉션이 이미 존재함 (즉시 검색 가능).
            - False: 컬렉션이 존재하지 않아 큐에 등록되었음 (백그라운드 처리 필요).

    Side Effects:
        - 컬렉션이 없을 경우, enqueue_request()을 호출하여 작업을 SQLite 큐에 등록합니다.

    Raises:
        AmbiguousFrameworkNameException을 : 만약 프레임워크 이름이 애매모호할 경우 (ex, spring) AmbiguousFrameworkNameException을 raises합니다.
        이 경우 더 구체적인 프레임워크 이름을 다시 찾아야 합니다.
        InvalidFrameworkNameException : 프레임워크 이름이 특수문자로만 이루어진 유효하지 않은 문자열일 경우 raises합니다. 이 경우에도 프레임워크 이름을 다시 찾아야 합니다.

    """
    
    collection_nm = norm(framework)
    if collection_exists(collection_nm):
        return True
    enqueue_request(collection_nm, seed_url, question_id) # 큐 저장
    return False

async def crawl_and_embed(framework: str, seed_url: str) -> str:
    semaphore = asyncio.Semaphore(5)
    responses = await _crawl_docs(seed_url, semaphore)
    embed_docs(framework, responses)

# 검색
@tool
def similarity_search(collection_name, question):
    """
    chroma db의 collection_name 컬렉션에서 유저의 질문으로 유사도 검색을 한 결과를 반환합니다.
    score 기반으로 유사도가 0.75 이상인 경우를 우선적으로 반환하며
    만약 유사도가 0.75 이상인 경우가 없다면 전체 중 최대 5개를 반환합니다.
    결과는 아래 형식의 배열입니다.
    [
        {
            "url": "url",
            "txt": "txt"
        }
    ]
    """
    collection_nm = norm(collection_name)
    db_doc = get_chroma_collection(collection_nm)

    refs = []
    docs_with_score = db_doc.similarity_search_with_score(question)
    filtered = [doc for doc, score in docs_with_score if score >= 0.75 ]
    if not filtered:
        filtered = [doc for doc, _ in docs_with_score[:5]]
    for doc in filtered:
        refs.append({
            "url": doc.metadata.get("source", ""),
            "txt": summarizes_text(doc.page_content)
        })
    return refs

def load_prompt_from_txt(name: str, **kwargs) -> str:
    base_dir = Path(__file__).resolve().parent
    prompt_path = base_dir / "prompt" / name
    with open(prompt_path, encoding="utf-8") as f:
        template = f.read()
    return template.format(**kwargs)
