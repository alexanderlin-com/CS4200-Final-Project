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

# ---------------------------------------------------------
# VECTOR STORE / RETRIEVER SETUP
# ---------------------------------------------------------
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("PINECONE_INDEX_NAME")
if not index_name:
    st.error("PINECONE_INDEX_NAME not set in .env")
    st.stop()

index = pc.Index(index_name)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=os.environ.get("OPENAI_API_KEY"),
)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# base retriever for normal questions
base_retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8, "fetch_k": 24, "lambda_mult": 0.4},
)

# ---------------------------------------------------------
# QUESTION TYPE HELPERS
# ---------------------------------------------------------
def is_global_question(q: str) -> bool:
    q_lower = q.lower()
    keywords = [
        "plot summary",
        "in depth plot",
        "in-depth plot",
        "overall plot",
        "full plot",
        "summarize the story",
        "summarise the story",
        "tell me the story",
        "what happens in",
        "overall summary",
        "whole story",
    ]
    return any(k in q_lower for k in keywords)

def build_retrieval_query(current_prompt: str) -> str:
    """
    For follow-up questions like 'who are they', include the previous
    user message to give the retriever more semantic signal.
    """
    last_user = None
    # look back through history for the previous HumanMessage
    for msg in reversed(st.session_state.messages[:-1]):  # exclude current prompt
        if isinstance(msg, HumanMessage):
            last_user = msg.content
            break

    if last_user:
        return last_user + "\n\nFollow-up question: " + current_prompt
    return current_prompt

def get_context_docs(prompt: str):
    """
    For normal questions → use base retriever.
    For global questions → pull a lot of chunks and order by chunk_index.
    """
    if not is_global_question(prompt):
        # conversation-aware retrieval query
        retrieval_query = build_retrieval_query(prompt)
        return base_retriever.invoke(retrieval_query)

    # GLOBAL SUMMARY MODE
    # pull many chunks across the corpus
    big_k = 80
    # using empty string as query just samples broadly;
    # we care more about coverage than specific match
    docs = vector_store.similarity_search("", k=big_k)

    # sort them into story order using chunk_index
    def sort_key(d):
        meta = d.metadata or {}
        return meta.get("chunk_index", 0)

    docs_sorted = sorted(docs, key=sort_key)
    return docs_sorted

# ---------------------------------------------------------
# CHAT STATE SETUP
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
    # system message kept internal
    st.session_state.messages.append(
        SystemMessage(
            "You are a lore archivist for the fictional world of 'Me and the Boys'. "
            "You strictly answer using the provided story context. "
            "If the context does not contain the answer, say you don't know instead "
            "of inventing new facts."
        )
    )
    # first visible AI message
    st.session_state.messages.append(
        AIMessage(
            "Welcome to the **Me and the Boys Lore Engine**.\n\n"
            "Ask me about characters, battles, magic systems, factions, or events from the story."
        )
    )

# render chat history
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# ---------------------------------------------------------
# USER INPUT
# ---------------------------------------------------------
prompt = st.chat_input("Ask about the story...")

if prompt:
    # append the new user message to state
    st.session_state.messages.append(HumanMessage(prompt))

    with st.chat_message("user"):
        st.markdown(prompt)

    # retrieve docs (conversation-aware / global-aware)
    docs = get_context_docs(prompt)

    if not docs:
        reply_text = (
            "I couldn't find anything about that in the current story corpus. "
            "Either it's not written yet, or it's outside the ingested material."
        )
        with st.chat_message("assistant"):
            st.markdown(reply_text)
        st.session_state.messages.append(AIMessage(reply_text))
    else:
        # -------------------------------------------------
        # BUILD CONTEXT STRING
        # -------------------------------------------------
        context_blocks = []
        for i, d in enumerate(docs, 1):
            meta = d.metadata or {}
            filename = meta.get("filename", meta.get("source", "unknown"))
            idx = meta.get("chunk_index", "?")
            context_blocks.append(
                f"[DOC {i} | chunk_index={idx} | {filename}]\n{d.page_content}"
            )
        context_str = "\n\n---\n\n".join(context_blocks)

        # -------------------------------------------------
        # SYSTEM PROMPT FOR LLM
        # -------------------------------------------------
        if is_global_question(prompt):
            instruction = (
                "You are summarizing the entire plot of 'Me and the Boys'.\n"
                "You are given excerpts ordered roughly from the beginning to the end of the story.\n"
                "Write a coherent, in-depth plot summary that covers the main arcs from start to finish.\n"
                "If some transitions are missing, infer them conservatively from the given text.\n"
            )
        else:
            instruction = (
                "You are an in-universe historian and lorekeeper for the world of 'Me and the Boys'.\n"
                "Use ONLY the following story excerpts to answer the user's question. "
                "Do not contradict the text. Do not add lore that is not supported.\n"
            )

        system_prompt = (
            instruction
            + "\n\nStory context:\n"
            + context_str
            + "\n\nIf the context is ambiguous or missing, say you don't know."
        )

        # -------------------------------------------------
        # LLM CALL WITH RECENT CHAT HISTORY
        # -------------------------------------------------
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.4,
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

        # take last few conversational turns (excluding the *internal* system msg at index 0)
        history = [
            m for m in st.session_state.messages[1:]
            if isinstance(m, (HumanMessage, AIMessage))
        ]
        # keep only last 6 turns to avoid overloading context
        history = history[-6:]

        messages_for_llm = [SystemMessage(content=system_prompt)] + history

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""

            for chunk in llm.stream(messages_for_llm):
                delta = getattr(chunk, "content", "") or ""
                if delta:
                    full_response += delta
                    placeholder.markdown(full_response)

            # show sample of sources
            st.markdown("\n\n**Sample of sources used:**")
            for i, d in enumerate(docs[:10], 1):
                meta = d.metadata or {}
                src = meta.get("filename", meta.get("source", "unknown source"))
                idx = meta.get("chunk_index", "?")
                st.markdown(f"- DOC {i}: `{src}` (chunk_index={idx})")

        st.session_state.messages.append(AIMessage(full_response))
