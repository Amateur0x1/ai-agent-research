# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## High-Level Architecture

This repository contains multiple AI agent research projects organized as separate directories:

### Repository Structure
- **`openai-agents-python-main/`**: OpenAI Agents SDK - Python framework for building multi-agent workflows with providers support for 100+ LLMs
- **`ag-ui/`**: Event-based protocol that standardizes agent-user interactions with TypeScript and Python SDKs
- **`ai/`**: Vercel AI SDK - TypeScript toolkit for building AI-powered streaming user interfaces
- **`langgraph/`**: LangChain's graph-based framework for building stateful, multi-actor agent applications
- **`adk-python/`**: Agent Development Kit (Python) - Additional agent development tools

### Core Concepts Across Projects
- **Multi-Agent Systems**: All projects support coordination between multiple AI agents
- **Provider Agnostic**: Support for multiple LLM providers (OpenAI, Anthropic, Google, etc.)
- **Streaming Support**: Real-time streaming responses and events
- **State Management**: Persistent conversation history and agent state
- **Tool Integration**: Function calling and external tool usage

## Common Development Commands

### OpenAI Agents SDK (Python)
```bash
cd openai-agents-python-main

# Install dependencies using uv
make sync

# Run all checks (format, lint, mypy, tests)
make check

# Individual operations
make tests         # Run test suite
make lint          # Run linting
make mypy          # Run type checking
make format        # Format code
make format-check  # Check formatting

# Documentation
make build-docs    # Build documentation
make serve-docs    # Serve docs locally
```

### AG-UI (TypeScript/Python)
```bash
cd ag-ui

# TypeScript SDK
cd typescript-sdk
pnpm install       # Install dependencies
pnpm build         # Build all packages
pnpm dev           # Development mode
pnpm lint          # Run linting
pnpm test          # Run tests
pnpm format        # Format code

# Python SDK
cd python-sdk
poetry install     # Install dependencies
python -m unittest discover tests  # Run tests
poetry build       # Build distribution
```

### LangGraph (Python)
```bash
cd langgraph

# Install and run checks for all libraries
make all           # Lint, format, lock, test all projects
make install       # Install dependencies for all projects
make test          # Run tests for all projects
make lint          # Lint all projects
make format        # Format all projects
```

### AI SDK (TypeScript)
```bash
cd ai

# Install dependencies
pnpm install

# Build packages
pnpm build

# Run tests
pnpm test

# Development
pnpm dev

# Linting and formatting
pnpm lint
pnpm format
```

## Project-Specific Architecture

### OpenAI Agents SDK
- **Agent Loop**: Continuous execution with LLM calls, tool processing, and handoffs
- **Handoffs**: Specialized tool calls for transferring control between agents
- **Sessions**: Automatic conversation history management using SQLite or Redis
- **Tracing**: Built-in tracking with support for external processors (Logfire, AgentOps, etc.)

### AG-UI Protocol
- **Event-Driven**: All communication through typed events (BaseEvent subtypes)
- **Transport Agnostic**: SSE, WebSockets, HTTP binary support
- **Observable Pattern**: RxJS Observables for streaming responses
- **State Management**: STATE_SNAPSHOT and STATE_DELTA with JSON Patch

### LangGraph
- **Graph-Based**: Define agent workflows as computational graphs
- **Stateful**: Built-in state management and persistence
- **Checkpointing**: Save and restore execution state at any point
- **Human-in-the-Loop**: Support for human intervention in agent workflows

### Testing and Quality Assurance
- All projects use comprehensive testing with pytest (Python) or Jest/Vitest (TypeScript)
- Type checking with mypy (Python) and TypeScript compiler
- Linting with ruff (Python) and ESLint (TypeScript)
- Code formatting with ruff (Python) and Prettier (TypeScript)

## Development Workflow
- Each project maintains independent versioning and release cycles
- Use `uv` for Python dependency management where available
- Use `pnpm` for TypeScript/Node.js dependency management
- Integration tests demonstrate cross-framework compatibility
- Examples directories provide working demonstrations of key features