# AI-Powered Customer Support Automation System

This project implements a LangGraph-based customer support automation system for ABC Technologies. It accepts customer queries, classifies the issue, routes the request to the correct department, retrieves relevant company document text, uses ChatOllama to generate responses, stores conversation memory in SQLite, and handles high-risk requests with human approval.

##gitHub Link: https://github.com/seshadri-07/AI-Powered-Customer-Support-Automation-System.git

## Requirements Covered

1. Accept customer queries.
2. Identify the customer issue type.
3. Route the query to the correct support department.
4. Retrieve relevant information from company documents.
5. Remember previous customer interactions using SQLite.
6. Escalate critical requests to a human supervisor.
7. Generate the final customer response.
8. Export the workflow diagram as PDF/SVG, with optional PNG/JPG.
9. Demonstrate the system using five predefined sample queries.
10. Allow extra user queries from the console.

## Support Departments

| Department | Handles |
| --- | --- |
| Sales | Product information, pricing plans, subscription plans |
| Technical Support | Application errors, upload crashes, login/configuration issues |
| Billing | Invoice, payment, refund, and billing questions |
| Account | Password reset, profile update, activation/deactivation |
| Memory Recall | Previous customer issue lookup from SQLite |

## High-Risk Requests

These requests require human supervisor approval:

- Refund requests
- Subscription cancellation
- Account closure requests
- Compensation requests
- Escalation to management

In predefined demo mode, approval is simulated automatically. In user input mode, the program asks for supervisor approval in the console.

## Project Files

| File | Purpose |
| --- | --- |
| `customer.py` | Main LangGraph customer support automation program |
| `customer_memory.sqlite3` | SQLite memory database generated when the program runs |
| `workflow_diagram.png` | Optional workflow image if Pillow is installed |
| `requriments.txt` | Dependency list file |

## Installation Steps

1. Create and activate a virtual environment.

```powershell
mkdir customer_support
cd customer_support
python -m venv .venv
.venv\Scripts\activate.ps1
```

2. Install required packages.

```powershell
pip install rich langgraph langchain-ollama
```

3. Install and start Ollama, then pull the model used by the code.

```powershell
pip install ollama
ollama pull qwen2.5:3b
```
check weather ollama is running or not  http://localhost:11434.
Make sure Ollama is running before executing the Python file.

## How to Run

Run the main program:

```powershell
python customer.py
```

The program will:

1. Reset/create the SQLite memory database.
2. Export the workflow diagram.
3. Run the five predefined demo queries.
4. Start user input mode.

To stop user input mode, type:

```text
exit
```

You can also type:

```text
quit
q
thank you
```

## Predefined Demo Queries

| Query | Expected Path |
| --- | --- |
| What are the pricing plans available for your software? | Sales |
| I forgot my account password. | Account |
| My application crashes whenever I upload a file. | Technical Support |
| I need a refund for my annual subscription. | Billing with human approval |
| What was my previous support issue? | Memory Recall |

## Workflow

The LangGraph workflow is:

```text
START
  -> classify_intent
      -> memory_node
      -> retrieve_docs
          -> sales_agent
          -> technical_agent
          -> billing_agent
          -> account_agent
              -> approval_node
                  -> supervisor_node
                      -> END
```

The same workflow is exported as:


- `workflow_diagram.png` 


## How It Works

1. `classify_intent` checks the query and assigns an intent: `sales`, `technical`, `billing`, `account`, or `memory`.
2. `retrieve_docs` selects the matching company document text.
3. The correct department agent calls `ChatOllama` and asks it to answer using only the retrieved document text.
4. `approval_node` checks high-risk requests and asks for supervisor approval during interactive mode.
5. `supervisor_node` finalizes the response and saves the interaction to SQLite memory.
6. `memory_node` answers previous issue questions from the SQLite database.

## Notes

- The code uses `ChatOllama(model="qwen2.5:3b")`.
- If Ollama is not running, response generation may fail.
- The SQLite database is recreated for each program run because `reset_demo_memory()` is called at startup.

## outputs
<img width="1224" height="749" alt="image" src="https://github.com/user-attachments/assets/3d76d27a-c16e-48c2-9e98-39bc0af220e6" />
<img width="1322" height="684" alt="image" src="https://github.com/user-attachments/assets/4fe82edb-6128-4d34-b1f6-7c9ae0b6a26c" />
<img width="1313" height="614" alt="image" src="https://github.com/user-attachments/assets/ffd01588-b2b2-4fc4-96d8-f3256f38c8f9" />
<img width="951" height="324" alt="image" src="https://github.com/user-attachments/assets/e0197887-8cd3-43ae-a6e8-30baa7a47b87" />
<img width="1305" height="699" alt="image" src="https://github.com/user-attachments/assets/47558f3e-abd8-4e10-af25-3b4715d2a495" />
<img width="966" height="337" alt="image" src="https://github.com/user-attachments/assets/1b1067b2-bbfa-4af4-9482-ac6bd8177e19" />

## Data base schema
<img width="1544" height="606" alt="image" src="https://github.com/user-attachments/assets/42de98c1-f18c-48db-97d1-53bbf1ee34b3" />




