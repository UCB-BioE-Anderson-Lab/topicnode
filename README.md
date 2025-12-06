topicnode
=========

Command line tool for maintaining a tree of topic nodes for a literature review.

Usage:

    python -m topicnode.cli <topic_root> <pdfs_path>

where <topic_root> contains:
- topic.txt
- pdfs/
- root/

All LLM calls are routed through openai_requests.ask_gpt, which must be available
on the Python path in the environment where this tool is run.

Installation
------------

1. Create a Python virtual environment:

    python3 -m venv .venv

2. Activate the virtual environment:

    source .venv/bin/activate

3. Install the dependencies:

    pip install -r requirements.txt

OpenAI API Key Configuration
----------------------------

Before running topicnode, create a `.env` file in the project root containing your OpenAI API key:

    OPENAI_API_KEY=sk-proj-0-hj3g

The tool `openai_requests` will automatically load this value so the LLM calls succeed. Ensure the `.env` file is not committed to version control.

Topic File Requirements
-----------------------

The topic_root directory must contain a file named topic.txt. This file should hold a plain‑language description of the review goal, including key questions, scope, and any constraints. The engine uses this text to guide initial characterization, feature selection, split creation, and summarization.
