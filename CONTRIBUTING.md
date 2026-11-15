# Contributing to HarmonyShield

Thank you for your interest in contributing to **HarmonyShield**! We welcome contributions to improve our audio forensics pipeline, expand generator generalization benchmarks, and enhance detection interfaces.

## Code of Conduct

Please maintain a collaborative, respectful, and scientifically honest environment. Report any evaluation metrics transparently, avoiding overfitting to in-domain sets.

## Development Setup

1. **Fork and Clone the Repository:**
   ```bash
   git clone https://github.com/ParamBande/HarmonyShiled.git
   cd HarmonyShiled
   ```

2. **Set Up Python Environment (Python 3.10–3.13):**
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Linux / macOS:
   source venv/bin/activate

   pip install -r requirements.txt
   ```

3. **Verify Model Weights:**
   Run the model verification script to ensure checkpoints are in place:
   ```bash
   python scripts/verify_models.py
   ```
   If checkpoints are missing, run:
   ```bash
   python scripts/download_checkpoints.py
   ```

4. **Frontend Setup (Node.js 18+):**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

## Running Tests

Before submitting a Pull Request, confirm all unit and integration tests pass:
```bash
pytest tests/
```
And verify that the frontend builds cleanly:
```bash
cd frontend && npm run build
```

## Pull Request Guidelines

- Ensure new code conforms to PEP 8 standards.
- Add unit tests for any new feature extraction, inference logic, or consensus policy.
- Document any new experiments in `experiments/` with full provenance and leakage audit notes.
- Do not commit binary model checkpoints (`*.pt`, `*.pth`) or raw copyrighted audio directly into Git. Use `models/model_manifest.json` and release asset attachments.
