from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from config import (
    OPENAI_MODEL_NAME,
    OPENAI_MODEL_TEMPERATURE,
)
from memory import get_session_history
from vectorstore import get_vectorstore
from prompts import contextualize_prompt, qa_prompt


def get_rag_chain():
    llm = ChatOpenAI(
        model=OPENAI_MODEL_NAME,
        temperature=OPENAI_MODEL_TEMPERATURE,
    )
    retriever = get_vectorstore().as_retriever()
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)
    question_answer_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=qa_prompt,
    )
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

def get_conversational_rag_chain():
    rag_chain = get_rag_chain()
    return RunnableWithMessageHistory(
        runnable=rag_chain,
        get_session_history=get_session_history,
        input_messages_key='input',
        history_messages_key='chat_history',
        output_messages_key='answer',
    )


_PERSONA_SYSTEM_PROMPT = (
    'Você é o atendente da Bora Treinar Suplementos. Seu tom é energético e usa emojis de academia '
    '(💪🏋️\u200d♂️🔥) sem exagerar. '
    'Você recebe uma INSTRUÇÃO DO SISTEMA dizendo exatamente o que deve comunicar ao cliente. '
    'Nunca invente preços ou produtos que não estejam na instrução. '
    'Seja conciso, amigável e focado em fechar a venda.'
)


def generate_persona_response(instruction: str, user_message: str, session_id: str) -> str:
    """Generates a persona-styled response via OpenAI, reading and saving to Redis chat history."""
    llm = ChatOpenAI(
        model=OPENAI_MODEL_NAME,
        temperature=0.7,
    )
    history = get_session_history(session_id)
    system_content = f'{_PERSONA_SYSTEM_PROMPT}\n\nINSTRUÇÃO DO SISTEMA: {instruction}'
    messages = [SystemMessage(content=system_content)]
    messages += list(history.messages)[-10:]
    messages.append(HumanMessage(content=user_message))
    response = llm.invoke(messages)
    history.add_user_message(user_message)
    history.add_ai_message(response.content)
    return response.content


def invoke_rag_chain(user_message: str, session_id: str) -> str:
    """Invokes the conversational RAG chain (saves history automatically)."""
    chain = get_conversational_rag_chain()
    result = chain.invoke(
        {'input': user_message},
        config={'configurable': {'session_id': session_id}},
    )
    return result.get('answer', '')
