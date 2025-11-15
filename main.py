from dotenv import load_dotenv
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from typing import List
from web_operations import serp_search, reddit_search_api

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
    return {"selected_reddit_urls": []}

def retrieve_reddit_posts(state: State) -> State:
    return {"reddit_post_data": ""}

def analyze_google_results(state: State) -> State:
    return {"google_analysis": ""}

def analyze_bing_results(state: State) -> State:
    return {"bing_analysis": ""}

def analyze_reddit_results(state: State) -> State:
    return {"reddit_analysis": ""}

def synthesize_analyses(state: State) -> State:
    return {"final_answer": ""}


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
