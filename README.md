# ZBench + SageMath Integration

[![Python](https://img.shields.io/badge/python-3.13+-blue.svg)](https://python.org)
[![SageMath](https://img.shields.io/badge/sagemath-9.x+-green.svg)](https://sagemath.org)
[![ZBench](https://img.shields.io/badge/zbench-enhanced-orange.svg)](https://github.com/zeroentropy-ai/zbench)
[![Mathematical Engine](https://img.shields.io/badge/math-exact%20computation-purple.svg)](#)
[![Reliability](https://img.shields.io/badge/convergence-100%25-brightgreen.svg)](#)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Enhanced ZBench pipeline with SageMath mathematical backend for reliable Elo calculations and document ranking.

## Quick Start

### 1. Setup Environment
```bash
cd zbench
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

### 2. Run Document Ranking
```bash
python -m zbench.annotation_with_sage ../sample_dataset.jsonl output.jsonl
```

## Key Features

- **100% Convergence Reliability** - SageMath backend handles edge cases vanilla ZBench fails on
- **Comprehensive Logging** - Detailed mathematical analysis in `sage/logs/`
- **Both Statistical Models** - Bradley-Terry and Thurstone implementations
- **Enhanced Diagnostics** - Numerical stability warnings and quality assessment

## Architecture

- `sage/` - SageMath mathematical engine with exact computation
- `zbench/` - Enhanced ZBench pipeline with SageMath integration
- `zbench/compare-original-zbench/` - Comparative analysis vs vanilla ZBench

## Requirements

- Python 3.13+
- SageMath 9.x+
- API keys for OpenAI/Anthropic (in `zbench/.env`)

See `zbench/results.md` for detailed performance analysis and `zbench/compare-original-zbench/` for mathematical comparison.