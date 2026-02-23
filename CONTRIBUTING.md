# Contributing to QMDZvec

Thanks for your interest in improving agent memory! 🧠

## Quick Start

```bash
git clone https://github.com/yoniassia/QMDZvec.git
cd QMDZvec
pip install zvec numpy pytest

# Start the server
python3.10 zvec/server.py &

# Run tests
python3.10 -m pytest tests/ -v
```

## Development Workflow

1. **Fork** the repo and create a feature branch
2. **Write tests** for your changes (see `tests/`)
3. **Run the full test suite** — all 34 tests must pass
4. **Submit a PR** with a clear description

## What We'd Love Help With

- **Embedding integrations** — Support for more embedding models beyond `embeddinggemma-300m`
- **Neo4j knowledge graph** — Entity extraction → graph relationships
- **Better chunking** — Smarter markdown chunking strategies
- **Fleet Memory** — WebSocket subscriptions, memory consensus
- **Performance** — Benchmarks, optimizations, caching

## Code Style

- Python 3.10+
- Type hints where practical
- Docstrings on public functions
- Keep it simple — this runs inside AI agents, reliability > cleverness

## Testing

```bash
# Start zvec server first
python3.10 zvec/server.py &

# Run all tests
python3.10 -m pytest tests/ -v

# Run specific test file
python3.10 -m pytest tests/test_zvec_server.py -v
```

All tests must pass before merging. Server tests require a running Zvec instance on port 4010.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
