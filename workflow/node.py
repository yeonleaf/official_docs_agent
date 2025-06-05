from typing import TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from enum import Enum
import json
from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv
import streamlit as st
from workflow.tools import get_general_keyword_docs, embed_general_docs, get_official_url, embed_docs, similarity_search, verify_framework, load_prompt_from_txt
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
import uuid
from langgraph.graph import StateGraph, START, END
from langchain.agents import AgentExecutor
import re
# from sentence_transformers import SentenceTransformer, util
from config.llm_client import llm, call_llm_with_backoff, embeddings
from config.constants import MAX_RETRIES
from sklearn.metrics.pairwise import cosine_similarity

class AgentType:
    START= "start"
    REF_GENERATOR="ref_generator"
    FRAMEWORK_REF="framework_ref"
    GENERAL_REF="general_ref"
    REF_CRITIC="ref_critic"
    PARAPHRASE="paraphrase"
    ANSWER_GENERATOR="answer_generator"
    ANSWER_CRITIC="answer_critic"
    END= "end"

class ReferenceType(str, Enum):
    FRAMEWORK="framework"
    GENERAL="general"

class Reference(TypedDict):
    url: str
    typ: ReferenceType
    txt: str

class AgentState(TypedDict):
    question: str
    question_id: str
    paraphrased_question: str
    references: List[Reference]
    route: str
    answer: str
    error: str
    ref_retries: int
    ans_retries: int

class NodeMessage(str, Enum):
    INVALID_QUESTION="기술과 연관성이 없는 질문입니다."
    EXCEED_MAX_TRIAL="최대 재시도 횟수를 초과했습니다."
    ENQUEUE_WORK="현재 참조 가능한 프레임워크 레퍼런스가 없습니다. 백그라운드에서 작업 진행중입니다. 사이드바의 Processing 탭에서 작업 처리 현황을 확인하세요."
    FAIL_JSON_PARSING="json 파싱에 실패했습니다. llm이 유효한 json 응답을 생성하지 않았습니다."

def json_parse(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 실패 시 fallback 프롬프트
        fallback_prompt = f"""
        아래 응답은 JSON 파싱에 실패했습니다. 올바른 JSON 배열로 수정해 주세요:

        {text}
        """
        fixed = llm.invoke(fallback_prompt).content
        return json.loads(fixed)

def reference_node(state: AgentState) -> AgentState:
    system_prompt = "너는 사용자의 질문이 기술에 대한 질문인지 판단하는 AI야."
    formatted_reference = load_prompt_from_txt("reference.txt", question=state['question'], paraphrased_question=state['paraphrased_question'])
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=formatted_reference)
    ]
    
    response = call_llm_with_backoff(messages)
    if response.content == "Y":
        return {
                **state,
                "route": AgentType.FRAMEWORK_REF,
                "references": []
            }
    else:
        return {
            **state,
            "route": END,
            "answer": response.content,
            "error": NodeMessage.INVALID_QUESTION
        }

tools_for_framework = [get_official_url, verify_framework, similarity_search]

def framework_node(state: AgentState) -> AgentState:
    react_in  = {"messages": []}
    try:
        formatted_framework = load_prompt_from_txt("framework.txt", question=state["question"], question_id=state["question_id"], paraphrased_question=state["paraphrased_question"])
        framework_agent = create_react_agent(llm, tools=tools_for_framework, prompt = formatted_framework)
        react_out = framework_agent.invoke(react_in)
        try:
            output = react_out["messages"][-1].content
            parsed = json_parse(output)

            # success가 "Y"인 항목만 필터링
            successful_refs = [
                {"url": p["url"], "txt": p["txt"], "typ": ReferenceType.FRAMEWORK, "success": p["success"]}
                for p in parsed if p.get("success") == "Y"
            ]

            # 성공한 항목이 하나도 없다면 종료
            if not successful_refs:
                return {
                    **state,
                    "error": NodeMessage.ENQUEUE_WORK,
                    "route": END
                }
            ret = {
                **state,
                "references": state["references"] + successful_refs,
                "route": AgentType.GENERAL_REF,
            }
            return ret
        except Exception as e:
            if state["ref_retries"] < MAX_RETRIES:
                return {
                    **state,
                    "route": AgentType.REF_GENERATOR,
                    "ref_retries": state["ref_retries"]+1,
                    "references": []
                }
            return {
                **state,
                "route": END,
                "error": NodeMessage.FAIL_JSON_PARSING
            }
    except Exception as e:
        st.error(f"ReAct 에이전트 예외: {e}")
    
tools_for_general = [get_general_keyword_docs, embed_general_docs, similarity_search]

def general_node(state: AgentState) -> AgentState:
    react_in  = {"messages": []}
    
    formatted_general = load_prompt_from_txt("general.txt", question=state["question"], question_id=state["question_id"], paraphrased_question=state["paraphrased_question"])
    general_agent = create_react_agent(llm, tools=tools_for_general, prompt=formatted_general)
    react_out = general_agent.invoke(react_in)
    output = react_out["messages"][-1].content
    try:
        parsed = json_parse(output)
        ret = {
            **state,                       
            "references": state["references"] + [
                {"url": p["url"], "txt": p["txt"], "typ": ReferenceType.GENERAL}
                for p in parsed
            ],
            "route": AgentType.REF_CRITIC,
        }
        return ret
    except Exception as e:
        if state["ref_retries"] < MAX_RETRIES:
            return {
                **state,
                "route": AgentType.GENERAL_REF,
                "ref_retries": state["ref_retries"]+1,
                "references": []
            }
        return {
            **state,
            "route": END,
            "error": NodeMessage.FAIL_JSON_PARSING
        }


def ref_critic_node(state: AgentState) -> AgentState:
    question = state["question"]
    paraphrased_question = state["paraphrased_question"]
    question_id = state["question_id"]
    references = state["references"]
    if len(references) == 0:
        if state["ref_retries"] < MAX_RETRIES:
            return {
                **state,
                "ref_retries": state["ref_retries"]+1,
                "route": AgentType.REF_GENERATOR,
            }
        else:
            return {
                **state,
                "route": END,
                "error": NodeMessage.EXCEED_MAX_TRIAL
            }
    system_prompt = "너는 기술 문서 품질 평가 AI야."
    formatted_prompt = load_prompt_from_txt("ref_critic.txt", question=question, question_id=question_id, paraphrased_question=paraphrased_question)
    for idx, ref in enumerate(state["references"]):
        formatted_prompt += f"({idx})\n{ref['txt']}\n"
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=formatted_prompt)
    ]
    
    response = call_llm_with_backoff(messages)

    try:
        result = json_parse(response.content)
        
        correct_count = sum(1 for ans in result if str(ans).strip().upper().startswith("Y"))
        
        ratio = correct_count / len(references)

        if correct_count >= 1:
            return {
                **state,
                "route": AgentType.ANSWER_GENERATOR
            }
        else:
            if (state["ref_retries"] < MAX_RETRIES):
                return {
                    **state,
                    "route": AgentType.PARAPHRASE,
                    "references": [],
                    "ref_retries": state["ref_retries"]+1
                }
            return {
                    **state,
                    "route": END,
                    "error": NodeMessage.EXCEED_MAX_TRIAL
                }
    except Exception as e:
        if state["ref_retries"] < MAX_RETRIES:
            return {
                **state,
                "route": AgentType.PARAPHRASE,
                "ref_retries": state["ref_retries"]+1,
                "references": []
            }
        return {
            **state,
            "route": END,
            "error": NodeMessage.FAIL_JSON_PARSING
        }

def paraphrase_node(state: AgentState) -> AgentState:
    question = state["question"]
    system_prompt = "당신은 검색 쿼리 최적화 AI입니다."
    formatted_prompt = load_prompt_from_txt("paraphrase.txt", question=question)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=formatted_prompt)
    ]
    response = call_llm_with_backoff(messages)
    paraphrased_question = response.content.strip()

    # question과 paraphrased_question의 유사도 구하기
    embedding_1 = embeddings.embed_query(question)
    embedding_2 = embeddings.embed_query(paraphrased_question)
    similarity_score = cosine_similarity([embedding_1], [embedding_2])[0][0]
    print(repr(question))
    print(repr(paraphrased_question))
    print(similarity_score)
    return {
        **state,
        "paraphrased_question": "" if similarity_score < 0.7 else paraphrased_question,
        "route": AgentType.REF_GENERATOR
    }

def generate_answer_node(state: AgentState) -> AgentState:
    question = state["question"]
    paraphrased_question = state["paraphrased_question"]
    references = state["references"]

    # 1. 레퍼런스 요약 프롬프트 작성
    reference_prompts = []
    for ref in references:
        prompt = f"""
        문서:\n{ref['txt']}\n
        출처:\n{ref['url']}
        """
        reference_prompts.append(prompt)
    
    # 2. 최종 답변 프롬프트 작성
    context_block = "\n".join(f"- {s}" for s in reference_prompts)
    system_prompt = """
    당신은 사용자가 입력한 질문에 대해 공식적인 출처를 기반으로 답변하는 AI입니다.
    """
    formatted_prompt = load_prompt_from_txt("answer_generator.txt", question=question, paraphrased_question=paraphrased_question, context_block=context_block)

    # 3. 답변 생성
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=formatted_prompt)
    ]
    response = call_llm_with_backoff(messages)
    answer = response.content.strip()
    return {
        **state,
        "answer":   answer,
        "route":    AgentType.ANSWER_CRITIC,
    }

def answer_critic_node(state: AgentState) -> AgentState:
    formatted_prompt = load_prompt_from_txt("answer_critic.txt", question=state['question'], paraphrased_question=state["paraphrased_question"], answer=state['answer'])
    response = call_llm_with_backoff([HumanMessage(content=formatted_prompt)])
    answer = response.content.strip()

    new_retry_cnt = state.get("ans_retries", 0) if answer == "Y" else state.get("ans_retries", 0) + 1
    retries_over = new_retry_cnt >= MAX_RETRIES

    return {
        **state,
        "route": END if answer == "Y" or retries_over else AgentType.REF_GENERATOR,
        "ans_retries":  new_retry_cnt,
    }