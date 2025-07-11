# Smart Ticket Triage Agent

A weekend starter project that showcases **Pydantic AI + spaCy + Hugging Face** for intelligent customer support ticket triage.

## 🎯 What it does

Transforms raw customer support emails into structured, validated JSON that your CRM or help-desk software can route automatically.

**Input:** Free-form email text  
**Output:** Validated JSON with customer ID, product, sentiment, urgency, entities, summary, and next action

## 🚀 Quick Start

### 1. Setup Environment

**Using UV (Recommended)**
```bash
# Install UV if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and create virtual environment
uv sync

# Download spaCy model
uv run python -m spacy download en_core_web_trf
```

**Using pip (Alternative)**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_trf
```

### 2. Run the Demo

**Using UV**
```bash
# Run the demo (no API keys required)
uv run python test_demo.py

# Or process a specific email
uv run python triage.py sample_emails.txt

# Or pipe email text
echo "I'm having trouble with Widget-X login" | uv run python triage.py
```

**Using pip**
```bash
# Run the demo (no API keys required)
python test_demo.py

# Or process a specific email
python triage.py sample_emails.txt

# Or pipe email text
echo "I'm having trouble with Widget-X login" | python triage.py
```

## 🏗️ Architecture

```
┌─────────────────┐
│  Email Input    │
└─────────┬───────┘
          ▼
┌─────────────────┐      ┌──────────────────────┐
│  spaCy NER      │◄────►│  Hugging Face        │
│  Entity Extract │      │  Sentiment Analysis  │
└─────────┬───────┘      └──────────┬───────────┘
          ▼                         ▼
          ┌─────────────────────────────────────┐
          │     PydanticAI Agent (LLM)          │
          │     Structured JSON Output          │
          └─────────────────────────────────────┘
```

## 📁 Project Structure

```
├── models.py          # Pydantic schemas (Ticket, Entity)
├── extractors.py      # ML components (spaCy NER, HF sentiment)
├── agent.py           # PydanticAI agent orchestration
├── triage.py          # CLI interface
├── test_demo.py       # Demo without API keys
├── sample_emails.txt  # Sample data
├── requirements.txt   # Dependencies (legacy)
└── pyproject.toml     # UV/modern Python dependencies
```

## 🛠️ Usage Examples

### CLI Usage

**Using UV**
```bash
# Process single email
uv run python triage.py email.txt

# Output as JSON
uv run python triage.py email.txt --json

# Use different model
uv run python triage.py email.txt --model gpt-4
```

**Using pip**
```bash
# Process single email
python triage.py email.txt

# Output as JSON
python triage.py email.txt --json

# Use different model
python triage.py email.txt --model gpt-4
```

### Programmatic Usage

```python
from agent import TriageAgent

agent = TriageAgent()
ticket = agent.process_email_sync(email_text)
print(ticket.dict())
```

## 📊 Sample Output

```json
{
  "customer_id": "C_SARAH_JOHNSON",
  "product": "Widget-X",
  "sentiment": "negative",
  "urgency": "high",
  "entities": [
    {
      "text": "Widget-X",
      "label": "PRODUCT",
      "start": 8,
      "end": 16,
      "confidence": 1.0
    }
  ],
  "summary": "Critical system outage affecting multiple customers",
  "next_action": "escalate_to_tier_2",
  "confidence_score": 0.89
}
```

## 🔧 Configuration

### Environment Variables

```bash
# Required for full LLM functionality
export OPENAI_API_KEY="your-api-key"

# Optional: Use different models
export PYDANTIC_AI_MODEL="gpt-4o-mini"
```

### Model Configuration

- **spaCy**: `en_core_web_trf` (transformer-based NER)
- **Sentiment**: `distilbert-base-uncased-finetuned-sst-2-english`
- **LLM**: `gpt-4o-mini` (configurable)

## 📈 Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Schema Validation | ≥99% | Pydantic guarantees |
| Entity Precision | >85% | spaCy transformer model |
| Sentiment Accuracy | >90% | DistilBERT fine-tuned |
| Latency | <3s CPU | Full pipeline |

## 🛡️ Security Notes

- No PII is logged by default
- API keys are environment variables only
- Consider using local models for sensitive data

## 💡 Common Messages

### "Device set to use mps:0"
This message appears on Apple Silicon Macs (M1/M2/M3) and indicates that PyTorch is using **Metal Performance Shaders** for GPU acceleration. This is **good** - it means your models are running faster! 

To suppress this message, you can:
- Ignore it (recommended - it's just informational)
- Set `PYTORCH_ENABLE_MPS_FALLBACK=1` to force CPU usage
- Use `2>/dev/null` to redirect stderr: `python triage.py --demo 2>/dev/null`

## 🔄 Extensions

- **Multilingual**: Add language detection and models
- **Fine-tuning**: Train spaCy on your domain data
- **Local LLM**: Replace OpenAI with Ollama/Llama
- **Batch Processing**: Add queue-based processing
- **Web API**: FastAPI wrapper for HTTP endpoints

## 🎓 Learning Outcomes

This project teaches:
- **Pydantic AI**: Schema-first LLM agents
- **spaCy**: Industrial-strength NLP
- **Hugging Face**: Pre-trained transformer models
- **Integration**: Combining multiple ML libraries
- **Validation**: Structured output guarantees

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

---

**Built with ❤️ as a weekend project to demonstrate the power of modern NLP stacks!**