# chatbot_rag.py
import os
from dotenv import load_dotenv

import streamlit as st
from pinecone import Pinecone

from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

st.title("Me and the Boys — Lore Engine")

# --- Vector store / retriever setup ---
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("PINECONE_INDEX_NAME")
index = pc.Index(index_name)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=os.environ.get("OPENAI_API_KEY"),
)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 12, "lambda_mult": 0.5},
)

# --- Chat history setup ---
if "messages" not in st.session_state:
    st.session_state.messages = []

    # Internal instruction
    st.session_state.messages.append(
        SystemMessage(
            "You are a lore archivist for the fictional world of 'Me and the Boys'. "
            "You strictly answer using the provided story context. "
            "If the context does not contain the answer, say you don't know instead "
            "of inventing new facts."
        )
    )

    # First visible message
    st.session_state.messages.append(
        AIMessage(
            "Welcome to the **Me and the Boys Lore Engine**.\n\n"
            "Ask me about characters, battles, magic systems, factions, or events from the story."
        )
    )

# Render history
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Input bar
prompt = st.chat_input("Ask about the story...")

if prompt:
    # show user message
    st.session_state.messages.append(HumanMessage(prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # retrieve context
    docs = retriever.invoke(prompt)

    if not docs:
        # no retrieval hits → no bullshit
        reply_text = (
            "I couldn't find anything about that in the current story corpus. "
            "Either it's not written yet, or it's outside the ingested material."
        )
        with st.chat_message("assistant"):
            st.markdown(reply_text)
        st.session_state.messages.append(AIMessage(reply_text))
    else:
        # format context
        context_blocks = []
        for i, d in enumerate(docs, 1):
            meta = d.metadata or {}
            label = meta.get("source_category", "story")
            context_blocks.append(
                f"[DOC {i} | {label}]\n{d.page_content}"
            )
        context_str = "\n\n---\n\n".join(context_blocks)

        system_prompt = (
            "You are an in-universe historian and lorekeeper for the world of 'Me and the Boys'.\n"
            "Use ONLY the following story excerpts to answer the user's question. "
            "Do not contradict the text. Do not add lore that is not supported.\n\n"
            f"Story context:\n{context_str}\n\n"
            "When you answer:\n"
            "- Be concise but specific.\n"
            "- If relevant, reference characters, locations, or events by name.\n"
            "- If the context is ambiguous or missing, say you don't know.\n"
        )

        messages_for_llm = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ]

        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.4,
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            for chunk in llm.stream(messages_for_llm):
                delta = getattr(chunk, "content", "") or ""
                if delta:
                    full_response += delta
                    message_placeholder.markdown(full_response)

            # after answer, show sources
            st.markdown("\n\n**Sources used:**")
            for i, d in enumerate(docs, 1):
                meta = d.metadata or {}
                src = meta.get("source", "unknown source")
                st.markdown(f"- DOC {i}: `{src}`")

        st.session_state.messages.append(AIMessage(full_response))
