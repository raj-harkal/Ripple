# Ripple: Open-Source AI Change-Impact Explorer

**Ripple** is a static Python change-impact analyzer and interactive explorer built for fast, trustworthy code review and refactoring. Given a Python repository and a target function or method, Ripple determines its statically discoverable callers, transitive impact paths, and candidate tests, and pairs them with a **Gemma 4** explanation strictly grounded in source evidence.

---

## The Problem

When modifying a function in an unfamiliar or growing Python codebase, answering *"What could break if I change this?"* is surprisingly difficult:
1. **Text search (`grep`) is shallow:** It misses cross-module import aliases, multi-hop transitive callers, and which end-to-end tests actually exercise a low-level helper.
2. **Pure LLM analysis is prone to hallucination:** Asking an ungrounded LLM to trace callers across files often results in invented call paths, missed dependencies, or false confidence in complete coverage.
3. **Dynamic analysis requires code execution:** Running or importing untrusted repositories just to inspect their call graph poses security risks and environment setup overhead.

---

## The Solution

Ripple separates **deterministic static facts** from **AI-inferred hypotheses**:

1. **Safe Static AST Parsing:** Ripple ingests Python repositories using Python's built-in `ast.parse`—never importing or executing analyzed code—and extracts functions, classes, methods, line ranges, import bindings, and call sites.
2. **Directed Call Graph & Reverse Reachability:** Using `NetworkX`, Ripple builds an explicit directed graph (`caller -> callee` with `relationship="CALLS"`) and traverses incoming edges backward from a selected target function to compute 1-hop direct callers, multi-hop transitive callers (strictly excluding the changed function itself), and reachable `test_*` functions.
3. **Explicit Unresolved Call Reporting:** Whenever Ripple encounters dynamic invocation primitives (`getattr`, `eval`, `exec`, `__import__`), wildcard imports (`from mod import *`), or complex chained expressions, it records them as **unresolved calls** rather than guessing or claiming 100% coverage.
4. **Validated Gemma 4 Explanations:** Ripple sends bounded source excerpts and verified graph edges to **Gemma 4**. A post-generation validator cross-checks every model-claimed relationship against the `NetworkX` graph, strips any fabricated edges, and visually separates **Verified Static Facts** from **Inferred Risks & Test Suggestions**.

---

## Key Features

- **Zero Code Execution:** Analyzes repositories purely as text and AST trees; enforces strict file count, byte size, symlink, and path-traversal guardrails.
- **Transitive Impact & Path Tracing:** Computes all reverse-reachable callers and displays the exact `Caller -> ... -> Target` dependency chains alongside file and line evidence.
- **Candidate Test Discovery:** Automatically highlights `test_*` functions and test modules that statically reach the modified symbol.
- **Hallucination Guardrail for Gemma 4:** Validates JSON responses from Gemma 4 against the ground-truth AST graph and flags any rejected edges or out-of-scope symbols.
- **Offline Fallback Synthesizer:** Works out-of-the-box without an API key or internet connection using a deterministic offline evidence synthesizer.
- **Built-in Sample Project & Verification Suite:** Ships with an embedded 5-file sample repository (`billing`, `api`, `tests`) and `pytest` suite so you can test and demo immediately.

---

## Tech Stack

- **Language:** Python 3.9+
- **Static Analysis:** Python standard library `ast`
- **Graph Engine:** `NetworkX` (`nx.DiGraph`)
- **Frontend UI:** `Streamlit`
- **AI Inference:** **Gemma 4** (`google/gemma-4-26b-a4b-it:free` via OpenRouter or Google Generative Language API)
- **Testing:** `pytest`

---

## Setup & Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Compilation & Environment

```bash
python3 -m py_compile ripple.py
python3 ripple.py --check-env
```

### 3. Configure Gemma 4 (OpenRouter / Environment Variables)

Ripple reads your inference configuration from environment variables (and also allows live overrides in the Streamlit sidebar):

```bash
export GEMMA_API_BASE="https://openrouter.ai/api/v1"
export GEMMA_MODEL="google/gemma-4-26b-a4b-it:free"
export GEMMA_API_KEY="your_openrouter_api_key_here"
```

> **Security Best Practice:** Always provide `GEMMA_API_KEY` via environment variables or a local `.env` file excluded by `.gitignore`. Never commit live API keys to version control. If `GEMMA_API_KEY` is unset, Ripple automatically uses its deterministic `EvidenceGroundedFallbackProvider`.

### 4. Run the Unit Tests

Run the built-in verification suite or execute via `pytest`:

```bash
python3 ripple.py
pytest ripple.py -v
```

### 5. Launch the Interactive Streamlit UI

```bash
streamlit run ripple.py
```

Inside the browser UI, you can:
- Switch between the **Built-in Sample Project** and any **Local Directory Path**.
- Select a target function or method (e.g., `billing.tax.get_tax_rate`) to inspect its source excerpt, direct callers, transitive impact paths, candidate tests, and supporting graph edges.
- View the **Gemma 4 Grounded Change-Impact Explanation** and expand the **Unresolved / Dynamic Calls** table.

### 6. Helpful CLI Commands

```bash
# Run an end-to-end JSON demo on the built-in sample project
python3 ripple.py --demo

# Write the built-in 5-file sample repository to a directory on disk
python3 ripple.py --materialize-sample ./sample_repo
```

---

## Known Limitations

Ripple is intentionally scoped as an honest static analyzer for a narrow subset of Python:

1. **Dynamic Dispatch & Reflection:** Calls made via `getattr()`, `eval()`, `exec()`, `__import__()`, dictionary dispatch tables, higher-order callbacks, or decorators are not resolved into graph edges. Instead, Ripple logs them in the **Unresolved Calls** report so reviewers know where static visibility ends.
2. **Polymorphism & Complex Attribute Chains:** Method resolution supports `self.method()` within the enclosing class, direct function imports, module-aliased calls (`mod.fn()`), and class `__init__` constructors. Deeply chained calls (e.g., `a().b().c()`) or complex multi-inheritance method resolution order (MRO) overrides are reported as unresolved.
3. **Wildcard Imports:** Statements like `from module import *` obscure static symbol bindings and are flagged as unresolved warnings.
4. **Repository Size Guardrails:** By default, `RepoLimits` caps ingestion at **200 Python files**, **256 KB per file**, and **5 MB total** (skipping `.git`, `venv`, `node_modules`, and symlinks) to keep analysis fast and bounded.
5. **Single-Process Architecture:** Ripple runs entirely in-process without external databases, vector stores, or background workers.