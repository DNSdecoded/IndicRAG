# Contributing to IndicRAG

Thank you for your interest in contributing to IndicRAG! This project aims to make scientific research accessible across Indian languages through advanced multilingual RAG capabilities.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Git
- Basic understanding of RAG systems and NLP

### Setup Instructions

1. **Fork the repository**
   - Visit [https://github.com/DNSdecoded/IndicRAG](https://github.com/DNSdecoded/IndicRAG)
   - Click "Fork" in the top right

2. **Clone your fork**
   ```bash
   git clone https://github.com/your-username/IndicRAG.git
   cd IndicRAG
   ```

3. **Set up development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate  # Windows
   
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys (Groq, etc.)
   ```

5. **Verify installation**
   ```bash
   python test_pipeline.py
   python example_query.py
   ```

## 🛠️ Development Workflow

### 1. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
# Examples:
# - feature/add-tamil-support
# - fix/embedding-dimension-mismatch
# - docs/update-api-examples
```

### 2. Make Your Changes
- Write clean, well-documented code
- Follow the existing code style and structure
- Add inline comments for complex logic
- Update relevant documentation

### 3. Test Thoroughly
```bash
# Run test pipeline
python test_pipeline.py

# Test with different languages
python example_query.py --lang hi  # Hindi
python example_query.py --lang bn  # Bengali
python example_query.py --lang ta  # Tamil

# Test API endpoints (if running locally)
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "आर्टिफिशियल इंटेलिजेंस क्या है?", "language": "hi"}'
```

### 4. Commit Your Changes
```bash
git add .
git commit -m "feat: add Tamil language support with Indic tokenizer"
# Use conventional commits: feat, fix, docs, style, refactor, test, chore
```

### 5. Submit a Pull Request
- Push to your fork: `git push origin feature/your-feature-name`
- Open a PR on GitHub
- Provide a clear description of changes
- Reference related issues (e.g., "Closes #42")
- Wait for review and address feedback

## 🎯 Priority Areas for Contribution

### 🔥 High Priority
- **Indian Language Support**: Add support for additional Indic languages (Tamil, Telugu, Kannada, Malayalam, Punjabi, Odia, etc.)
- **Translation Quality**: Improve translation accuracy for technical/scientific terms
- **Embedding Models**: Test and compare performance of different multilingual embedding models (Indic-BERT variants, MuRIL, etc.)
- **Performance Optimization**: Reduce query latency, implement caching strategies
- **Bug Fixes**: Fix any reported issues, especially language-specific edge cases

### 🚀 Medium Priority
- **Advanced Reranking**: Implement cross-encoder models for better result ranking
- **Citation Extraction**: Automatically extract and format citations from research papers
- **Query Expansion**: Add query reformulation for better retrieval
- **Code-Switching Support**: Handle mixed language queries (e.g., Hinglish)
- **Document Preprocessing**: Better handling of scientific notation, equations, tables

### 💡 Nice to Have
- **Additional Formats**: Support for HTML, EPUB, XML documents
- **Kubernetes Deployment**: Production-ready K8s templates
- **Monitoring**: Dashboards for query performance and system health
- **Load Testing**: Scripts to test system under load
- **Voice Integration**: Support for voice queries in Indian languages
- **Mobile App**: Native mobile application

## 📝 Code Guidelines

### Python Style
- Follow [PEP 8](https://pep8.org/) style guide
- Use type hints for function signatures
- Write docstrings (Google style preferred)
- Keep functions focused (single responsibility)
- Maximum line length: 100 characters

Example:
```python
def translate_query(query: str, target_lang: str) -> str:
    """
    Translate user query to target language.
    
    Args:
        query: Input query string
        target_lang: ISO 639-1 language code (e.g., 'hi', 'bn')
    
    Returns:
        Translated query string
    
    Raises:
        ValueError: If target_lang is not supported
    """
    # Implementation here
    pass
```

### Documentation
- Update `README.md` when adding user-facing features
- Update `ARCHITECTURE.md` for system design changes
- Add examples in `examples/` directory
- Document API changes in docstrings
- Include language-specific considerations

### Testing
- Test with at least 3 different Indian languages
- Verify both English and Indic script queries
- Check edge cases (empty queries, special characters, code-switching)
- Test with different document types (PDF, TXT, scientific papers)
- Verify Docker/API deployments work

### Language Support Checklist
When adding a new language:
- [ ] Add language code to supported languages list
- [ ] Test translation quality with technical terms
- [ ] Verify embedding model handles the script
- [ ] Test with real scientific documents
- [ ] Add examples in `examples/` directory
- [ ] Update documentation

## 🐛 Reporting Issues

When reporting issues, please include:

### For Bugs
- **Description**: Clear explanation of the problem
- **Steps to Reproduce**: Detailed steps to recreate the issue
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Environment**: OS, Python version, relevant dependencies
- **Language**: Which language(s) are affected
- **Logs/Screenshots**: Any error messages or visual evidence

### For Feature Requests
- **Use Case**: Why this feature is needed
- **Proposed Solution**: How you envision it working
- **Alternatives**: Other approaches you've considered
- **Impact**: Who benefits and how

### For Questions
- **Check Documentation**: Review README.md and ARCHITECTURE.md first
- **Search Issues**: See if already asked/answered
- **Be Specific**: Include context and what you've tried

## 🤝 Community

- Be respectful and inclusive
- Help others in discussions
- Share your use cases and feedback
- Star the repo if you find it useful! ⭐

## 📄 License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).

---

**Thank you for helping make scientific research accessible across Indian languages! 🇮🇳**

*Questions? Open an issue or reach out to the maintainers.*
