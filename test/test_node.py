import pytest
from unittest.mock import patch, MagicMock
import sys
import pathlib
from langgraph.graph import START, END

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from workflow.node import (
    reference_node,
    framework_node,
    general_node,
    ref_critic_node,
    paraphrase_node,
    answer_critic_node,
    json_parse,
    AgentType,
    NodeMessage,
    AgentState,
)

# ---------- 공통 더미/헬퍼 -------------------------------------------------
class DummyMsg:
    def __init__(self, content): self.content = content

BASE_STATE: AgentState = {
    "question": "question",
    "question_id": "uuid-1234",
    "paraphrased_question": "",
    "references": [],
    "route": "",
    "answer": "",
    "error": "",
    "ref_retries": 0,
    "ans_retries": 0,
}

# 1) reference_node
@patch("workflow.node.call_llm_with_backoff")
def test_reference_valid(mock_llm):
    mock_llm.return_value = DummyMsg("Y")
    s = reference_node(BASE_STATE)
    assert s["route"] == AgentType.FRAMEWORK_REF

@patch("workflow.node.call_llm_with_backoff")
def test_reference_invalid(mock_llm):
    mock_llm.return_value = DummyMsg("N")
    s = reference_node(BASE_STATE)
    assert s["route"] == END and s["error"] == NodeMessage.INVALID_QUESTION

# 2) framework_node
FW_PROMPT = "dummy fw prompt"

@patch("workflow.node.json_parse", return_value=[{"url":"u","txt":"t","success":"Y"}])
@patch("workflow.node.load_prompt_from_txt", return_value=FW_PROMPT)
@patch("workflow.node.create_react_agent")
def test_framework_success(mock_agent, prompt_mock, parse_mock):
    mock = MagicMock()
    mock.invoke.return_value = {"messages": [DummyMsg(""), DummyMsg("does not matter")]}
    mock_agent.return_value = mock

    s = framework_node(BASE_STATE)
    assert s["route"] == AgentType.GENERAL_REF
    assert len(s["references"]) == 1
    
@patch("workflow.node.load_prompt_from_txt", return_value=FW_PROMPT)
@patch("workflow.node.create_react_agent")
def test_framework_no_success_refs(_, mock_agent):
    mock = MagicMock()
    mock.invoke.return_value = {"messages":[DummyMsg(""), DummyMsg('[{"url":"u","txt":"t","success":"N"}]')]}
    mock_agent.return_value = mock
    s = framework_node(BASE_STATE)
    assert s["route"] == AgentType.REF_GENERATOR

@patch("workflow.node.MAX_RETRIES", 1)   # 빠른 테스트용
@patch("workflow.node.load_prompt_from_txt", return_value=FW_PROMPT)
@patch("workflow.node.create_react_agent")
def test_framework_json_fail_retry(_, mock_agent):
    mock = MagicMock()
    mock.invoke.return_value = {"messages":[DummyMsg(""), DummyMsg('INVALID_JSON')]}
    mock_agent.return_value = mock
    state = {**BASE_STATE, "ref_retries":0}
    s = framework_node(state)
    assert s["route"] == AgentType.REF_GENERATOR and s["ref_retries"] == 1

# 3) general_node
GEN_PROMPT = "dummy gen prompt"

@patch("workflow.node.json_parse")
@patch("workflow.node.load_prompt_from_txt", return_value=GEN_PROMPT)
@patch("workflow.node.create_react_agent")
def test_general_success(json_agent, _, mock_agent):
    mock = MagicMock()
    mock.invoke.return_value = {"messages":[DummyMsg(""), DummyMsg('[{"url":"g","txt":"doc","success":"Y"}]')]}
    mock_agent.return_value = mock
    json_agent = [
        {
            "url": "u",
            "txt": "t",
            "success": "Y"
        }
    ]
    prev = {**BASE_STATE, "route":AgentType.GENERAL_REF}
    s = general_node(prev)
    assert s["route"] == AgentType.REF_CRITIC

@patch("workflow.node.MAX_RETRIES", 1) 
@patch("workflow.node.load_prompt_from_txt", return_value=GEN_PROMPT) 
@patch("workflow.node.create_react_agent")  
def test_general_parse_fail_retry(mock_agent, _):
    mock = MagicMock()
    mock.invoke.return_value = {
        "messages": [DummyMsg(""), DummyMsg("INVALID")]  # json_parse 실패 유도
    }
    mock_agent.return_value = mock

    s = general_node(BASE_STATE)
    assert s["route"] == AgentType.GENERAL_REF and s["ref_retries"] == 1
    
# 4) ref_critic_node
CR_PROMPT = "critic prompt"

@patch("workflow.node.load_prompt_from_txt", return_value=CR_PROMPT)
@patch("workflow.node.call_llm_with_backoff")
def test_ref_critic_pass(mock_llm, _):
    mock_llm.return_value = DummyMsg('["Y"]')
    prev = {**BASE_STATE,
            "references":[{"url":"x","txt":"doc","typ":"framework"}]}
    s = ref_critic_node(prev)
    assert s["route"] == AgentType.ANSWER_GENERATOR

@patch("workflow.node.MAX_RETRIES", 2)
@patch("workflow.node.load_prompt_from_txt", return_value=CR_PROMPT)
@patch("workflow.node.call_llm_with_backoff")
def test_ref_critic_fail_retry(mock_llm, _):
    mock_llm.return_value = DummyMsg('["N"]')
    prev = {**BASE_STATE,
            "references":[{"url":"x","txt":"doc","typ":"framework"}],
            "ref_retries":1}
    s = ref_critic_node(prev)
    assert s["route"] == AgentType.PARAPHRASE and s["ref_retries"] == 2

@patch("workflow.node.load_prompt_from_txt", return_value=CR_PROMPT)
@patch("workflow.node.call_llm_with_backoff")
def test_ref_critic_no_refs(mock_llm, _):
    s = ref_critic_node(BASE_STATE)
    assert s["route"] == AgentType.REF_GENERATOR

# 5) paraphrase_node
@patch("workflow.node.load_prompt_from_txt", return_value="p prompt")
@patch("workflow.node.call_llm_with_backoff")
def test_paraphrase(mock_llm, _):
    mock_llm.return_value = DummyMsg("question")
    s = paraphrase_node(BASE_STATE)
    assert s["paraphrased_question"] == "question"
    assert s["route"] == AgentType.REF_GENERATOR

@patch("workflow.node.load_prompt_from_txt", return_value="p prompt")
@patch("workflow.node.call_llm_with_backoff")
def test_paraphrase(mock_llm, _):
    mock_llm.return_value = DummyMsg("something different")
    s = paraphrase_node(BASE_STATE)
    assert s["paraphrased_question"] == ""
    assert s["route"] == AgentType.REF_GENERATOR

# 6) answer_critic_node
AC_PROMPT = "critic ans"

@patch("workflow.node.load_prompt_from_txt", return_value=AC_PROMPT)
@patch("workflow.node.call_llm_with_backoff")
def test_answer_critic_pass(mock_llm, _):
    mock_llm.return_value = DummyMsg("Y")
    state = {**BASE_STATE, "answer":"foo"}
    s = answer_critic_node(state)
    assert s["route"] == END

@patch("workflow.node.MAX_RETRIES", 2)
@patch("workflow.node.load_prompt_from_txt", return_value=AC_PROMPT)
@patch("workflow.node.call_llm_with_backoff")
def test_answer_critic_retry_then_over(mock_llm, _max):
    # _max == 1 (쓰지 않을 거면 _ 로 둬도 됨)
    mock_llm.return_value = DummyMsg("N")

    state = {**BASE_STATE, "answer": "foo", "ans_retries": 0}

    s1 = answer_critic_node(state)
    assert s1["route"] == AgentType.REF_GENERATOR
    assert s1["ans_retries"] == 1

    s2 = answer_critic_node(s1)
    assert s2["route"] == END

# 7) json_parse fallback
@patch("workflow.node.llm")
def test_json_parse_fallback(mock_llm):
    mock_llm.invoke.return_value = DummyMsg('["fixed"]')
    broken = "[unclosed"
    assert json_parse(broken) == ["fixed"]