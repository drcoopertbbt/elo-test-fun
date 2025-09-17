# Simplified ZBench + SageMath Architecture
## Single File Interface Design

### Overview
The architecture has been simplified to use `main.sage` as the single source of truth for all SageMath calculations, with `main.py` as a clean Python interface. This creates a much cleaner and more maintainable design.

### Architecture Diagram

```
┌─────────────────────────────────────────────┐
│              ZBench Pipeline                │
│                                             │
│  Data Loading → Pair Generation → LLM Calls │
│                                             │
│           Preference Matrix W               │
│                    │                        │
└────────────────────┼────────────────────────┘
                     │
                Python Import
                     │
                     ▼
┌─────────────────────────────────────────────┐
│              sage/main.py                   │
│           (Python Interface)               │
│                                             │
│  • SageMathInterface class                  │
│  • calculate_elos_sage() function          │
│  • Temp file management                    │
│  • Error handling & fallback               │
│                    │                        │
│              subprocess call                │
│                    │                        │
└────────────────────┼────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│              sage/main.sage                 │
│        (Single Source of Truth)            │
│                                             │
│  • Core zELO algorithm                      │
│  • ZBench's exact implementation           │
│  • Service mode (JSON I/O)                 │
│  • Test suite                              │
│  • All mathematical functions              │
└─────────────────────────────────────────────┘
```

### File Structure

#### sage/main.sage
**The single source of truth for all SageMath calculations**
- Core `calculate_elos_zbench_exact()` function
- Service mode for JSON input/output
- Comprehensive test suite
- All mathematical implementations

```sage
# Can run standalone
sage main.sage                          # Run test suite
sage main.sage input.json output.json   # Service mode
```

#### sage/main.py  
**Clean Python interface layer**
- `SageMathInterface` class
- `calculate_elos_sage()` convenience function
- Automatic temp file management
- Error handling with fallback to ZBench's internal implementation

```python
from main import calculate_elos_sage
elos, status = calculate_elos_sage(preference_matrix, document_ids)
```

#### zbench/zbench/sage_engine.py
**ZBench integration module**
- `calculate_elos_with_sage_backend()` function
- Automatic fallback to internal implementation
- Status reporting

#### zbench/zbench/annotation_with_sage.py
**High-precision annotation pipeline**
- Drop-in replacement for `annotation.py`
- Uses SageMath for all Elo calculations
- Maintains full ZBench compatibility

### Usage Examples

#### 1. Standalone SageMath Testing
```bash
cd sage
sage main.sage                    # Run comprehensive test suite
python3 main.py --test           # Test Python interface
python3 main.py                  # Run demo calculation
```

#### 2. ZBench Integration
```bash
cd zbench
python -m zbench.annotation_with_sage dataset.jsonl output.jsonl
```

#### 3. Direct Python Usage
```python
import sys; sys.path.append('sage')
from main import calculate_elos_sage

# Calculate Elos from preference matrix
elos, status = calculate_elos_sage(preference_matrix, document_ids)
print(f"Converged: {status['converged']}, MSE: {status.get('mse', 'N/A')}")
```

### Key Benefits

1. **Single Source of Truth**: All SageMath code lives in `main.sage`
2. **Clean Interface**: `main.py` provides simple Python bridge
3. **Automatic Fallback**: Falls back to ZBench's implementation if SageMath unavailable
4. **Dual Mode**: `main.sage` works standalone or as a service
5. **Full Compatibility**: Drop-in replacement for ZBench's `calculate_elos`
6. **Easy Testing**: Both files can be tested independently

### Test Results

**SageMath Interface Test:**
```
✓ SageMath connection successful
```

**Demo Calculation:**
```
Document A: 1.455969
Document B: -0.728305  
Document C: -0.727664
Status: Converged=True, Iterations=6
```

**Comprehensive Test Suite:**
```
TEST 1: Basic Bradley-Terry
Converged at iteration 134
Elo scores: [7.95, 2.55, -2.55, -7.95]

TEST 2: Service Mode  
Converged: true, Iterations: 6
Final loss: 2.383
```

### Migration Path

1. **Current ZBench users**: Use `annotation_with_sage.py` instead of `annotation.py`
2. **SageMath development**: Work directly with `main.sage`
3. **Python integration**: Import `main.py` for programmatic access
4. **Testing**: Use either file's built-in test capabilities

This simplified architecture provides a clean separation of concerns while maintaining the high-precision mathematical capabilities that make the SageMath engine superior to ZBench's internal implementation.