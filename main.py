from dotenv import load_dotenv
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from typing import List
from web_operations import serp_search, reddit_search_api, reddit_post_retrieval
from prompts import (
    get_reddit_analysis_messages,
    get_google_analysis_messages,
    get_bing_analysis_messages,
    get_synthesis_messages,
    get_reddit_url_analysis_messages,
)

load_dotenv()

llm = init_chat_model("gpt-4o")

class State(TypedDict):
    messages: Annotated[List[str], add_messages]
    user_question: str | None
    google_results: str | None
    bing_results: str | None
    reddit_results: str | None
    selected_reddit_urls: list[str] | None
    reddit_post_data: str | None
    google_analysis: str | None
    bing_analysis: str | None
    reddit_analysis: str | None
    final_answer: str | None


class RedditURLAnalysis(BaseModel):
    selected_urls: List[str] = Field(description="List of Reddit URLs that contain valuable information for answering the user's question.")

def google_search(state: State) -> State:
    user_question = state.get("user_question", "")
    print(f"Performing Google search for: {user_question}")

    google_results = serp_search(user_question, engine="google")
    print(f"Google search results: {google_results}")

    return {"google_results": google_results}

def bing_search(state: State) -> State:
    user_question = state.get("user_question", "")
    print(f"Performing Bing search for: {user_question}")

    bing_results = serp_search(user_question, engine="bing")
    print(f"Bing search results: {bing_results}")

    return {"bing_results": bing_results}

def reddit_search(state: State) -> State:
    user_question = state.get("user_question", "")
    print(f"Performing Reddit search for: {user_question}")

    reddit_results = reddit_search_api(user_question)
    print(f"Reddit search results: {reddit_results}")

    return {"reddit_results": reddit_results}

def analyze_reddit_posts(state: State) -> State:
    user_question = state.get("user_question", "")
    reddit_results = state.get("reddit_results", "")

    if not reddit_results:
        return {"selected_reddit_urls": []}
    
    structured_llm = llm.with_structured_output(RedditURLAnalysis)
    messages = get_reddit_url_analysis_messages(user_question, reddit_results) 

    try:
        analysis = structured_llm.invoke(messages)
        selected_urls = analysis.selected_urls

        for i, url in enumerate(selected_urls, start=1):
            print(f"Selected Reddit URL {i}: {url}")

    except Exception as e:
        print(f"Error during Reddit URL analysis: {e}")
        selected_urls = []

    return {"selected_reddit_urls": selected_urls}

def retrieve_reddit_posts(state: State) -> State:
    selected_urls = state.get("selected_reddit_urls", [])

    if not selected_urls:
        return {"reddit_post_data": []}
    
    print(f"Processing {len(selected_urls)} Reddit URLs for post retrieval...")

    reddit_post_data = reddit_post_retrieval(selected_urls)

    if reddit_post_data:
        print("Successfully got {len(reddit_post_data)} posts")

    else:
        print("Failed to get post data")
        reddit_post_data = []

    print(reddit_post_data)
    return {"reddit_post_data": reddit_post_data}        

def analyze_google_results(state: State) -> State:
    print("Analyzing Google results...")
    user_question = state.get("user_question", "")
    google_results = state.get("google_results", "")
    messages = get_google_analysis_messages(user_question, google_results)
    reply = llm.invoke(messages)
    return {"google_analysis": reply}

def analyze_bing_results(state: State) -> State:
    print("Analyzing Bing results...")
    user_question = state.get("user_question", "")
    bing_results = state.get("bing_results", "")
    messages = get_bing_analysis_messages(user_question, bing_results)
    reply = llm.invoke(messages)
    return {"bing_analysis": reply}

def analyze_reddit_results(state: State) -> State:
    print("Analyzing Reddit results...")
    user_question = state.get("user_question", "")
    reddit_results = state.get("reddit_results", "")
    reddit_post_data = state.get("reddit_post_data", "")
    messages = get_reddit_analysis_messages(user_question, reddit_results, reddit_post_data)
    reply = llm.invoke(messages)
    return {"reddit_analysis": reply}

def synthesize_analyses(state: State) -> State:
    print("Combining all results together...")
    user_question = state.get("user_question", "")
    google_analysis = state.get("google_analysis", "")
    bing_analysis = state.get("bing_analysis", "")
    reddit_analysis = state.get("reddit_analysis", "")

    messages = get_synthesis_messages(
        user_question, google_analysis, bing_analysis, reddit_analysis
    )
    reply = llm.invoke(messages)
    final_answer = reply.content

    return {"final_answer": final_answer, "messages":[{"role":"assistant","content":final_answer}]} 


graph_builder = StateGraph(State)

graph_builder.add_node("google_search", google_search)
graph_builder.add_node("bing_search", bing_search)
graph_builder.add_node("reddit_search", reddit_search)
graph_builder.add_node("analyze_reddit_posts", analyze_reddit_posts)
graph_builder.add_node("analyze_reddit_results", analyze_reddit_results)
graph_builder.add_node("analyze_google_results", analyze_google_results)
graph_builder.add_node("analyze_bing_results", analyze_bing_results)
graph_builder.add_node("retrieve_reddit_posts", retrieve_reddit_posts)
graph_builder.add_node("synthesize_analyses", synthesize_analyses)

graph_builder.add_edge(START, end_key="google_search")
graph_builder.add_edge(START, end_key="bing_search")
graph_builder.add_edge(START, end_key="reddit_search")

graph_builder.add_edge(start_key="google_search", end_key="analyze_reddit_posts")
graph_builder.add_edge(start_key="bing_search", end_key="analyze_reddit_posts")
graph_builder.add_edge(start_key="reddit_search", end_key="analyze_reddit_posts")
graph_builder.add_edge(start_key="analyze_reddit_posts", end_key="retrieve_reddit_posts")

graph_builder.add_edge(start_key="retrieve_reddit_posts", end_key="analyze_reddit_results")
graph_builder.add_edge(start_key="retrieve_reddit_posts", end_key="analyze_google_results")
graph_builder.add_edge(start_key="retrieve_reddit_posts", end_key="analyze_bing_results")

graph_builder.add_edge(start_key="analyze_reddit_results", end_key="synthesize_analyses")
graph_builder.add_edge(start_key="analyze_google_results", end_key="synthesize_analyses")
graph_builder.add_edge(start_key="analyze_bing_results", end_key="synthesize_analyses")

graph_builder.add_edge(start_key="synthesize_analyses", end_key=END)

graph = graph_builder.compile()

def main():
    print("Multi-Source Research Agent")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("Enter your research question: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        state = {
            "messages": [{"role": "user", "content": user_input}],
            "user_question": user_input,
            "google_results": None,
            "bing_results": None,
            "reddit_results": None,
            "selected_reddit_urls": None,
            "reddit_post_data": None,
            "google_analysis": None,
            "bing_analysis": None,
            "reddit_analysis": None,
            "final_answer": None,
        }

        print("\n  Starting parallel research process...\n")
        print("Launching Google Search, Bing Search, and Reddit Search...\n")

        final_state = graph.invoke(state)
        if final_state.get("final_answer"):
            print(f"\nFinal Answer:\n {final_state.get('final_answer')}\n")
            print("-" * 80 + "\n")

        print("-" * 80 + "\n")

if __name__ == "__main__":
    main()
