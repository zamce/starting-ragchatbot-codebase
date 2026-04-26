# AGENTS.md

## Overview
This file provides guidance for AI coding agents to understand and contribute effectively to the RAG system codebase.

## Project Description
The Retrieval-Augmented Generation (RAG) system is designed to answer questions about course materials using semantic search and AI-powered responses. It includes a FastAPI backend, a simple web-based frontend, and ChromaDB for vector storage.

## Key Commands

### Build and Run
- **Quick Start**:
  ```bash
  chmod +x run.sh
  ./run.sh
  ```
- **Manual Start**:
  ```bash
  cd backend
  uv run uvicorn app:app --reload --port 8000
  ```

### Testing
- No specific test commands are defined yet. Add test instructions here if applicable.

## Architecture

### Backend
- **Framework**: FastAPI
- **Key Modules**:
  - `ai_generator.py`: Handles AI-powered response generation.
  - `document_processor.py`: Processes course materials for indexing or querying.
  - `rag_system.py`: Implements the core RAG logic.
  - `vector_store.py`: Manages vector storage and retrieval.
  - `app.py`: Entry point for the FastAPI application.

### Frontend
- **Files**:
  - `index.html`: Main user interface.
  - `script.js`: Handles client-side logic and API interactions.
  - `style.css`: Defines visual styling.

### Documentation
- **README.md**: Provides setup instructions and an overview of the project.
- **Course Scripts**: Located in `docs/`.

## Conventions
- Follow Python best practices for backend development.
- Use semantic versioning for project updates.
- Environment variables are managed via a `.env` file.

## Potential Pitfalls
- Ensure Python 3.13 or higher is installed.
- The Anthropic API key must be set in the `.env` file.
- Use `uv` for dependency management.

## Links
- [README.md](README.md)