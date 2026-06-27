"""
AI-Powered Customer Support Automation System for ABC Technologies.

Requirements covered:
1. LangGraph workflow
2. State structure
3. Intent classification
4. Conditional routing
5. Department agents
6. RAG from company documents
7. SQLite memory
8. Human approval for high-risk requests
9. Supervisor response validation
10. Demo with five predefined queries and user input

Run:
    python customer.py
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional, TypedDict
from rich.console import Console
from langchain_ollama import ChatOllama
from langchain_core.runnables.graph import MermaidDrawMethod
from langgraph.graph import END, StateGraph



console = Console()
llm = ChatOllama(model="qwen2.5:3b", temperature=0.3) 


# ==========================================
# Knowledge Base Documents for RAG
# ==========================================

DOCUMENTS = {
    "sales": """
Pricing Guide:
Starter Plan: $29 per user/month for small teams.
Professional Plan: $79 per user/month with automation, integrations,
advanced reporting, and priority support.
Enterprise Plan: custom pricing with SSO, activity logs, security controls,
and dedicated success management.
""",
    "technical": """
Technical Manual:
Upload crashes can happen because of unsupported file formats, large files,
browser cache problems, expired login sessions, or incorrect configuration.
Try a smaller supported file, clear cache, log in again, and collect the error ID.
""",
    "billing": """
Company Policy Document:
Refund requests for annual subscriptions require human supervisor approval.
Approved refunds are processed to the original payment method within
five to seven business days. Invoice and payment issues are handled by Billing.
""",
    "account": """
FAQ Document:
Password resets can be completed from the login screen using Forgot Password.
Profile updates, account activation, and account deactivation are handled by
Account Support.
""",
    "policy": """
Company Policy Document:
Refund requests, subscription cancellation, account closure requests,
compensation requests, and management escalations must be reviewed by a
human supervisor before the final response is sent.
""",
}


DEMO_QUERIES = [
    "What are the pricing plans available for your software?",
    "I forgot my account password.",
    "My application crashes whenever I upload a file.",
    "I need a refund for my annual subscription.",
    "What was my previous support issue?",
]


WORKFLOW_DIAGRAM = """
LangGraph Workflow

START
  |
  v
classify_intent
  |
  +-- memory ---------------------> memory_node
  |
  +-- sales/technical/billing/account
          |
          v
       retrieve_docs
          |
          v
    route_to_department
          |
          +--> sales_agent
          +--> technical_agent
          +--> billing_agent
          +--> account_agent
                    |
                    v
              approval_node
                    |
                    v
              supervisor_node
                    |
                    v
                   END
"""


# ==========================================
# State
# ==========================================

class SupportState(TypedDict, total=False):
    customer_id: str
    user_query: str
    intent: str
    department: str
    retrieved_docs: str
    response: str
    final_response: str
    requires_approval: bool
    approved: bool
    interactive: bool


# ==========================================
# SQLite Memory
# ==========================================

DB_FILE = "customer_memory.sqlite3"


def create_memory_table() -> None:
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT,
                query TEXT,
                intent TEXT,
                response TEXT,
                created_at TEXT
            )
            """
        )


def save_memory(customer_id: str, query: str, intent: str, response: str) -> None:
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            """
            INSERT INTO memory (customer_id, query, intent, response, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (customer_id, query, intent, response, datetime.now().isoformat(timespec="seconds")),
        )


def get_previous_issue(customer_id: str) -> Optional[str]:
    with sqlite3.connect(DB_FILE) as conn:
        row = conn.execute(
            """
            SELECT query, intent
            FROM memory
            WHERE customer_id = ? AND intent != 'memory'
            ORDER BY id DESC
            LIMIT 1
            """,
            (customer_id,),
        ).fetchone()

    if not row:
        return None
    return f"{row[0]} ({row[1].title()} issue)"


# ==========================================
# Helper Logic
# ==========================================

def needs_human_approval(query: str) -> bool:
    high_risk_words = [
        "refund",
        "cancel",
        "cancellation",
        "close account",
        "account closure",
        "compensation",
        "management",
        "escalate",
    ]
    query = query.lower()
    return any(word in query for word in high_risk_words)


def contains_any(query: str, words: List[str]) -> bool:
    return any(word in query for word in words)


def generate_response_with_ollama(department: str, query: str, document_text: str) -> str:
    prompt = f"""
You are an ABC Technologies {department} support agent.
Answer the customer's question using only the document text below.
If supervisor approval is required, clearly say that human approval is required.
Keep the response short, polite, and helpful.

Customer question:
{query}

Document text:
{document_text}
"""

    if llm is None:
        return f"{department} Response:\n{document_text.strip()}"

    try:
        reply = llm.invoke(prompt)
        return reply.content.strip()
    except Exception as error:
        return (
            f"{department} Response:\n{document_text.strip()}\n\n"
            f"(Ollama response unavailable: {error})"
        )


# ==========================================
# LangGraph Nodes
# ==========================================

def classify_intent(state: SupportState) -> SupportState:
    query = state["user_query"].lower()

    if contains_any(query, ["previous", "last issue", "earlier issue"]):
        intent = "memory"
    elif contains_any(query, ["price", "pricing", "plan"]):
        intent = "sales"
    elif contains_any(query, ["crash", "error", "upload", "install", "login", "configuration"]):
        intent = "technical"
    elif contains_any(query, ["refund", "invoice", "payment", "billing", "charge"]):
        intent = "billing"
    elif contains_any(query, ["password", "profile", "activate", "deactivate", "account"]):
        intent = "account"
    else:
        intent = "account"

    state["intent"] = intent
    state["requires_approval"] = needs_human_approval(query)
    state["approved"] = not state["requires_approval"]
    return state


def retrieve_docs(state: SupportState) -> SupportState:
    intent = state["intent"]
    state["retrieved_docs"] = DOCUMENTS.get(intent, DOCUMENTS["policy"])
    if state["requires_approval"]:
        state["retrieved_docs"] += "\n" + DOCUMENTS["policy"]
    return state


def sales_agent(state: SupportState) -> SupportState:
    state["department"] = "Sales"
    state["response"] = generate_response_with_ollama(
        state["department"],
        state["user_query"],
        state["retrieved_docs"],
    )
    return state


def technical_agent(state: SupportState) -> SupportState:
    state["department"] = "Technical Support"
    state["response"] = generate_response_with_ollama(
        state["department"],
        state["user_query"],
        state["retrieved_docs"],
    )
    return state


def billing_agent(state: SupportState) -> SupportState:
    state["department"] = "Billing"
    state["response"] = generate_response_with_ollama(
        state["department"],
        state["user_query"],
        state["retrieved_docs"],
    )
    return state


def account_agent(state: SupportState) -> SupportState:
    state["department"] = "Account"
    state["response"] = generate_response_with_ollama(
        state["department"],
        state["user_query"],
        state["retrieved_docs"],
    )
    return state


def memory_node(state: SupportState) -> SupportState:
    previous_issue = get_previous_issue(state["customer_id"])
    state["department"] = "Memory Recall"
    state["requires_approval"] = False
    state["approved"] = True

    if previous_issue:
        state["response"] = f"Your previous support issue was: {previous_issue}"
    else:
        state["response"] = "No previous support issue was found for this customer."
    return state


def approval_node(state: SupportState) -> SupportState:
    if not state["requires_approval"]:
        state["approved"] = True
        return state

    if state.get("interactive"):
        console.print("\n[bold yellow]Human Supervisor Approval Required[/bold yellow]")
        decision = console.input("Approve this request? (yes/no): ").strip().lower()
        state["approved"] = decision in {"yes", "y"}
    else:
        # Demo mode simulates supervisor approval so the sample can run unattended.
        state["approved"] = True

    return state


def supervisor_node(state: SupportState) -> SupportState:
    if state["requires_approval"] and not state["approved"]:
        final_response = "Request rejected by supervisor. A human agent will follow up."
    elif state["requires_approval"]:
        final_response = "Supervisor Approved\n\n" + state["response"]
    else:
        final_response = state["response"]

    state["final_response"] = final_response
    save_memory(
        state["customer_id"],
        state["user_query"],
        state["intent"],
        final_response,
    )
    return state


# ==========================================
# Routing
# ==========================================

def route_from_classifier(state: SupportState) -> str:
    if state["intent"] == "memory":
        return "memory"
    return "rag"


def route_to_agent(state: SupportState) -> str:
    return state["intent"]


# ==========================================
# Graph
# ==========================================

def build_graph():
    

    graph = StateGraph(SupportState)

    graph.add_node("classifier", classify_intent)
    graph.add_node("rag", retrieve_docs)
    graph.add_node("sales", sales_agent)
    graph.add_node("technical", technical_agent)
    graph.add_node("billing", billing_agent)
    graph.add_node("account", account_agent)
    graph.add_node("memory", memory_node)
    graph.add_node("approval", approval_node)
    graph.add_node("supervisor", supervisor_node)

    graph.set_entry_point("classifier")

    graph.add_conditional_edges(
        "classifier",
        route_from_classifier,
        {"memory": "memory", "rag": "rag"},
    )
    graph.add_conditional_edges(
        "rag",
        route_to_agent,
        {
            "sales": "sales",
            "technical": "technical",
            "billing": "billing",
            "account": "account",
        },
    )

    graph.add_edge("sales", "approval")
    graph.add_edge("technical", "approval")
    graph.add_edge("billing", "approval")
    graph.add_edge("account", "approval")
    graph.add_edge("memory", "supervisor")
    graph.add_edge("approval", "supervisor")
    graph.add_edge("supervisor", END)

    return graph.compile()


def run_without_langgraph(state: SupportState) -> SupportState:
    state = classify_intent(state)

    if route_from_classifier(state) == "memory":
        state = memory_node(state)
    else:
        state = retrieve_docs(state)
        agents = {
            "sales": sales_agent,
            "technical": technical_agent,
            "billing": billing_agent,
            "account": account_agent,
        }
        state = agents[state["intent"]](state)
        state = approval_node(state)

    return supervisor_node(state)


app = build_graph()
GRAPH = app


def draw_workflow_image(file_name: str = "workflow_diagram.png") -> None:
    try:
        image_data = app.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
        )
    except Exception:
        image_data = app.get_graph().draw_png()

    Path(file_name).write_bytes(image_data)
    console.print(f"Workflow image saved as {file_name}")


def display_workflow_image() -> None:
    from IPython.display import Image, display

    try:
        image_data = app.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
        )
    except Exception:
        image_data = app.get_graph().draw_png()

    display(Image(image_data))


# ==========================================
# Demo and User Input
# ==========================================

def reset_demo_memory() -> None:
    memory_file = Path(DB_FILE)
    if memory_file.exists():
        try:
            memory_file.unlink()
        except PermissionError:
            pass
    create_memory_table()


def run_query(query: str, customer_id: str = "customer-1001", interactive: bool = False) -> SupportState:
    state: SupportState = {
        "customer_id": customer_id,
        "user_query": query,
        "requires_approval": False,
        "approved": False,
        "interactive": interactive,
    }

    if GRAPH:
        return GRAPH.invoke(state)
    return run_without_langgraph(state)


def print_result(title: str, query: str, result: SupportState) -> None:
    console.print("\n" + "=" * 70)
    console.print(f"[bold]{title}:[/bold] {query}")
    console.print("-" * 70)
    console.print(f"Department: {result['department']}")
    console.print(f"Intent: {result['intent'].title()}")
    console.print(f"Human Approval Required: {result['requires_approval']}")
    console.print(f"Approved: {result['approved']}")
    console.print("\n[bold]Final Response:[/bold]")
    console.print(result["final_response"])
    console.print("=" * 70)


def run_demo() -> None:
    console.print(WORKFLOW_DIAGRAM)
    console.print("[bold]Predefined Query Demonstration[/bold]")

    for index, query in enumerate(DEMO_QUERIES, start=1):
        result = run_query(query, interactive=False)
        print_result(f" Query {index}", query, result)


def run_user_input() -> None:
    console.print("\n[bold]User Input Query Mode[/bold]")
    console.print("Type your own query, or type exit to stop.")

    while True:
        query = console.input("\nEnter your query: ").strip()

        if query.lower() in {"exit", "quit","thank you", "q"}:
            console.print("Exiting customer support.")
            break

        if not query:
            console.print("Please enter a valid query.")
            continue

        result = run_query(query, interactive=True)
        print_result("User Query", query, result)


if __name__ == "__main__":
    reset_demo_memory()
    draw_workflow_image()
    run_demo()
    run_user_input()
