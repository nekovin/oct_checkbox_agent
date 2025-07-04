#!/bin/bash

# OCR Checkbox Agent - Git Repository Initialization Script
# This script helps set up the Git repository with initial commit

set -e

echo "🚀 OCR Checkbox Agent - Git Repository Setup"
echo "============================================="

# Check if we're already in a git repository
if [ -d ".git" ]; then
    echo "⚠️  Git repository already exists!"
    read -p "Do you want to continue? This will add files to the existing repo. (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
else
    echo "📁 Initializing new Git repository..."
    git init
fi

# Configure git if not already configured
if ! git config user.name > /dev/null 2>&1; then
    echo "👤 Setting up Git user configuration..."
    read -p "Enter your name: " git_name
    read -p "Enter your email: " git_email
    git config user.name "$git_name"
    git config user.email "$git_email"
    echo "✅ Git user configuration set"
fi

# Add all files
echo "📦 Adding files to Git..."
git add .

# Check if there are any changes to commit
if git diff --cached --quiet; then
    echo "ℹ️  No changes to commit"
else
    echo "💾 Creating initial commit..."
    git commit -m "feat: initial release of OCR Checkbox Agent

- Complete PDF processing with dual backend support
- Advanced OCR integration using Tesseract
- Computer vision-based checkbox detection  
- Multiple output formats (CSV, Excel, JSON)
- Batch processing with parallel execution
- Form template system for reusable patterns
- Comprehensive configuration management
- Quality assurance with confidence scoring
- Complete test suite and documentation
- Production-ready with error handling

🎯 Ready for extraction of checkbox data from PDF forms!"
fi

echo ""
echo "🎉 Git repository setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Create a new repository on GitHub"
echo "2. Add the remote origin:"
echo "   git remote add origin https://github.com/your-username/ocr-checkbox-agent.git"
echo "3. Push to GitHub:"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "🔧 Optional: Set up GitHub repository settings"
echo "- Enable Issues and Discussions"
echo "- Add repository description and topics"
echo "- Configure branch protection rules"
echo "- Set up secrets for CI/CD (PYPI_API_TOKEN)"
echo ""
echo "📚 Repository includes:"
echo "- ✅ Complete source code with modular architecture"
echo "- ✅ Comprehensive test suite with CI/CD workflows"
echo "- ✅ Documentation (README, CONTRIBUTING, CHANGELOG)"
echo "- ✅ GitHub templates (issues, PRs, workflows)"
echo "- ✅ Package configuration (setup.py, requirements.txt)"
echo "- ✅ Examples and sample configurations"
echo ""
echo "🚀 Ready to publish and share your OCR Checkbox Agent!"

# Show repository status
echo ""
echo "📊 Repository Status:"
git status --short
echo ""
echo "📈 Repository Statistics:"
echo "Files: $(find . -type f -not -path './.git/*' | wc -l)"
echo "Lines of code: $(find . -name '*.py' -not -path './.git/*' -exec wc -l {} + | tail -1 | awk '{print $1}')"
echo "Test files: $(find . -name 'test_*.py' | wc -l)"
echo ""