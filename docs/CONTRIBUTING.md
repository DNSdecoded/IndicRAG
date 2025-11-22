# Contributing to Multilingual Scientific RAG

Thank you for your interest in contributing! This is a generic research tool designed for anyone to use.

## 🚀 Getting Started

1. **Fork the repository**
2. **Clone your fork**
   ```bash
   git clone https://github.com/your-username/multilingual-rag.git
   cd multilingual-rag
   ```
3. **Set up development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

## 🛠️ Development Workflow

1. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clean, documented code
   - Follow existing code style
   - Add tests if applicable

3. **Test your changes**
   ```bash
   python test_pipeline.py  # Run tests
   python example_query.py  # Test end-to-end
   ```

4. **Submit a pull request**
   - Describe your changes
   - Reference any related issues

## 🎯 Areas for Contribution

### High Priority
- **Additional language support** (Bengali, Gujarati improvements)
- **Performance optimization** (caching, batching)
- **Documentation improvements**
- **Bug fixes**

### Medium Priority
- **Advanced reranking** (cross-encoder models)
- **Citation extraction** from PDFs
- **Query expansion** techniques
- **More embedding models** comparison

### Nice to Have
- **Additional document formats** (HTML, EPUB, XML)
- **Kubernetes deployment** templates
- **Monitoring dashboards**
- **Load testing** scripts

## 📝 Code Guidelines

### Python Style
- Follow PEP 8
- Use type hints
- Write docstrings for functions
- Keep functions focused and small

### Documentation
- Update README.md if adding features
- Update ARCHITECTURE.md for design changes
- Add examples for new functionality

### Testing
- Test with multiple languages
- Verify API endpoints
- Check Docker deployment

## 🐛 Reporting Issues

Use GitHub Issues with:
- **Bug reports**: Steps to reproduce, expected vs actual behavior
- **Feature requests**: Use case, proposed solution
- **Questions**: Check docs first, then ask

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for helping make scientific research accessible in every language! 🌍**
