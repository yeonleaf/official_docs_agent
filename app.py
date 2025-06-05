import streamlit as st
from dotenv import load_dotenv
from workflow.graph import create_graph
from workflow.node import AgentState, NodeMessage
from db.repository import init_db, read_queue_by_status, select_requests_by_question_id_and_status
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
import uuid
import chromadb
from config.logging import setup_logging
from workflow.tools import search_ddgs
from config.chroma_client import client
from config.constants import MAX_RETRIES

def start_research():
    topic = st.session_state.ui_topic
    graph = create_graph()
    initial_state: AgentState = {
        "question": topic,
        "question_id": st.session_state.question_id,
        "paraphrased_question": "",
        "references": [],
        "route": "",
        "answer": "",
        "error": "",
        "ref_retries": 0,
        "ans_retries": 0
    }
    result = None
    final_node = None

    stage_docs = {
        "ref_generator": "질문과 관련된 레퍼런스를 찾는 중...",
        "framework_ref": "질문과 관련된 프레임워크의 공식 문서를 분석하는 중...",
        "general_ref": "질문과 관련된 개념 키워드의 신뢰도 높은 문서를 분석하는 중...",
        "ref_critic": "레퍼런스가 질문과 연관성이 있는지 이차 검증을 거치는 중...",
        "paraphrase": "재검색을 위해 질문을 의미는 같되 표현이 다르도록 재작성하는 중...",
        "answer_generator": "레퍼런스를 기반으로 답변을 작성하는 중...",
        "answer_critic": "생성된 답변이 질문에 대한 적합한 답변인지 확인하는 중..."
    }

    stream = graph.stream(initial_state)
    with st.status("분석 중.."):
        for i, event in enumerate(stream):
            current_node = list(event.keys())[0]
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"{stage_docs[current_node]}")
            with col2:
                st.badge(f"재시도 횟수: ({event[current_node]['ref_retries']}/{MAX_RETRIES})")
            result = event
            final_node = current_node

    if "answer_critic" in result and "answer" in result["answer_critic"]:
        st.markdown(result["answer_critic"]["answer"])
    else:
        if result[final_node]["error"] == NodeMessage.ENQUEUE_WORK:
            alreadyIn = False
            for item in st.session_state["questions"]:
                if item["question_id"] == result[final_node]["question_id"]:
                    alreadyIn = True
            if not alreadyIn:
                st.session_state["questions"].append({
                    "question": result[final_node]["question"],
                    "question_id": result[final_node]["question_id"]
                })
        st.error(result[final_node]["error"].value)
        if result[final_node]["answer"]:
            st.write(result[final_node]["answer"])

def click_prev_question(question_id, question_text):
    st.session_state["ui_topic"] = question_text
    st.session_state["question_id"] = question_id

    st.session_state["questions"] = [
        q for q in st.session_state["questions"] if q["question_id"] != question_id
    ]

    st.session_state["submit_from_prev_question"] = True
    return

def render_sidebar():
    with st.sidebar:
        tab_done, tab_processing = st.tabs(["Done", "Processing"])
        # 완료 리스트
        with tab_done:
            done_list = read_queue_by_status("done",[])
            for job_id, collection, url in done_list:
                st.badge(collection, icon=":material/check:", color="green")
        # 진행 중, 대기 리스트 
        with tab_processing:
            for item in st.session_state.questions:
                question = item["question"]
                question_id = item["question_id"]
                with st.container():
                    collections = select_requests_by_question_id_and_status(question_id, ["running", "pending"])
                    st.button(question, disabled=len(collections) > 0, on_click=click_prev_question, args=(question_id, question))
                    for collection, status, created_at in collections:
                        icon = "💡" if status == "running" else "⏱"
                        color = "orange" if status == "running" else "grey"
                        st.badge(collection, icon=icon, color=color)

def generate_ui():
    st.set_page_config(page_title="공식 문서 마스터")

    st.title("🤖 공식 문서 마스터")
    st.markdown(
        """
        - 이 애플리케이션은 사용자가 입력한 질문에 대해 알맞은 기술을 유추하여 **공식 문서를 기반으로 답변**합니다.
        """
    )

    if "questions" not in st.session_state:
        st.session_state["questions"] = []

    if "submit_from_prev_question" not in st.session_state:
        st.session_state["submit_from_prev_question"] = False

    render_sidebar()

    with st.form(key="question_form"):
        st.text_input("질문을 입력해주세요.", key="ui_topic")
        if st.form_submit_button(label="submit") or st.session_state.submit_from_prev_question:
            if st.session_state.submit_from_prev_question:
                st.session_state.submit_from_prev_question = False
            else:
                alreadyIn = False
                for item in st.session_state.questions:
                    if item["question"] == st.session_state.ui_topic:
                        alreadyIn = True
                        break
                if not alreadyIn:
                    st.session_state["question_id"] = str(uuid.uuid4())
            start_research()


if __name__ == '__main__':
    init_db()
    setup_logging()
    generate_ui()