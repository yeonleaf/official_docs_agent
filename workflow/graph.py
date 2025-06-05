from workflow.node import AgentState, AgentType, reference_node, framework_node, general_node, ref_critic_node, paraphrase_node, generate_answer_node, answer_critic_node, llm
from langgraph.graph import StateGraph, START, END
import streamlit as st

def create_graph() -> StateGraph:
    # 그래프 생성
    workflow = StateGraph(state_schema=AgentState)

    # 노드 추가
    workflow.add_node(AgentType.REF_GENERATOR, reference_node)
    workflow.add_node(AgentType.FRAMEWORK_REF, framework_node)
    workflow.add_node(AgentType.GENERAL_REF, general_node)
    workflow.add_node(AgentType.REF_CRITIC, ref_critic_node)
    workflow.add_node(AgentType.PARAPHRASE, paraphrase_node)
    workflow.add_node(AgentType.ANSWER_GENERATOR, generate_answer_node)
    workflow.add_node(AgentType.ANSWER_CRITIC, answer_critic_node)

    # edge 연결
    workflow.add_edge(START, AgentType.REF_GENERATOR)
    workflow.add_conditional_edges(AgentType.REF_GENERATOR, lambda s: s["route"], [AgentType.FRAMEWORK_REF, END])
    workflow.add_conditional_edges(AgentType.FRAMEWORK_REF, lambda s: s["route"], [END, AgentType.GENERAL_REF])
    workflow.add_conditional_edges(AgentType.GENERAL_REF, lambda s: s["route"], [AgentType.REF_CRITIC])
    workflow.add_conditional_edges(AgentType.REF_CRITIC, lambda s: s["route"], [AgentType.PARAPHRASE, AgentType.ANSWER_GENERATOR, END])
    workflow.add_conditional_edges(AgentType.PARAPHRASE, lambda s: s["route"], [AgentType.REF_GENERATOR])    
    workflow.add_conditional_edges(AgentType.ANSWER_GENERATOR, lambda s: s["route"], [AgentType.ANSWER_CRITIC])
    workflow.add_conditional_edges(AgentType.ANSWER_CRITIC, lambda s: s["route"], [END, AgentType.REF_GENERATOR])

    graph = workflow.compile()
    return graph