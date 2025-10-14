# Contributing to TeachSim

Thank you for your interest in contributing to TeachSim! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)
- [Feature Requests](#feature-requests)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please be respectful and constructive in all interactions.

### Our Standards

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/teaching_simulation.git
   cd teaching_simulation
   ```
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## How to Contribute

There are many ways to contribute to TeachSim:

- 🐛 **Report bugs** - Help us identify issues
- 💡 **Suggest features** - Propose new functionality
- 📝 **Improve documentation** - Make it easier for others to understand
- 🔧 **Fix bugs** - Submit patches for known issues
- ✨ **Add features** - Implement new capabilities
- 🧪 **Write tests** - Improve code coverage and reliability
- 📊 **Add experiments** - Contribute new experimental configurations

## Development Setup

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Git

### Installation

1. Install dependencies:
   ```bash
   cd teaching_simulation
   pip install -r requirements.txt
   ```

2. Verify installation:
   ```bash
   python main.py --mock_test --n_rounds 5
   ```

### Development Dependencies

For development, you may want to install additional tools:

```bash
pip install black flake8 pytest mypy
```

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) style guidelines with some modifications:

- **Line length**: Maximum 88 characters (Black default)
- **Indentation**: 4 spaces (no tabs)
- **Imports**: Organized in three groups (stdlib, third-party, local)
- **Naming conventions**:
  - Classes: `PascalCase`
  - Functions/methods: `snake_case`
  - Constants: `UPPER_CASE`
  - Private methods: `_leading_underscore`

### Code Formatting

We recommend using [Black](https://github.com/psf/black) for consistent code formatting:

```bash
black teaching_simulation/
```

### Type Hints

Use type hints for function parameters and return values:

```python
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray

def compute_likelihood(
    data: List[Point], 
    hypothesis: Hypothesis
) -> NDArray[np.float64]:
    ...
```

### Documentation

- **Docstrings**: Use Google-style docstrings for all public functions and classes
- **Comments**: Explain *why*, not *what* (code should be self-explanatory)
- **README**: Update documentation when adding new features

Example docstring:

```python
def select_data_point(self, strategy: str = "random") -> Tuple[Point, int]:
    """
    Select a data point to show to the student.
    
    Args:
        strategy: Selection strategy ('random' or 'hypothesis')
        
    Returns:
        A tuple of (selected_point, cluster_label)
        
    Raises:
        ValueError: If strategy is not recognized
    """
    ...
```

## Testing Guidelines

### Running Tests

Before submitting changes, ensure all tests pass:

```bash
# Run mock test
python main.py --mock_test --n_rounds 5

# Run with different configurations
python main.py --config configs/test.yaml
```

### Writing Tests

When adding new features:

1. Add test cases that cover normal operation
2. Add edge case tests
3. Test error handling
4. Verify backward compatibility

### Test Configuration

Create test configuration files in `configs/` directory:

```yaml
# configs/test_new_feature.yaml
n_hypotheses: 5
n_clusters: 2
teacher_strategy: hypothesis
student_mode: rational
...
```

## Submitting Changes

### Commit Messages

Write clear, descriptive commit messages:

```
[Component] Brief description of change

More detailed explanation if needed. Wrap at 72 characters.

- List specific changes
- Use bullet points for clarity
- Reference issue numbers (#123)
```

Examples:
```
[agents] Add uncertainty-based student strategy

[env] Fix data generation for uniform initialization

[docs] Update README with new parameter descriptions

[experiments] Add experiment 7 comparing belief particle counts
```

### Pull Request Process

1. **Update your fork** with the latest changes:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Push your changes**:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a Pull Request** on GitHub with:
   - Clear title describing the change
   - Description of what changed and why
   - Reference to related issues (if any)
   - Screenshots/plots if applicable
   - Test results

4. **Address review comments** promptly and respectfully

5. **Ensure CI passes** (if configured)

### PR Checklist

Before submitting, ensure:

- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] Tests pass locally
- [ ] Commit messages are clear
- [ ] No unnecessary files included
- [ ] Configuration files are valid YAML
- [ ] Results can be reproduced with provided configs

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

1. **Description**: Clear description of the issue
2. **Steps to reproduce**: Exact steps to trigger the bug
3. **Expected behavior**: What should happen
4. **Actual behavior**: What actually happens
5. **Environment**:
   - Python version
   - OS (macOS, Linux, Windows)
   - Package versions (`pip list`)
6. **Configuration**: Config file or command-line arguments used
7. **Logs/errors**: Full error messages and stack traces
8. **Results**: Any relevant output files or plots

### Issue Template

```markdown
## Bug Description
Brief description of the bug

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- Python version: 3.10
- OS: macOS 14
- NumPy version: 1.26.4

## Configuration
```yaml
# paste config here
```

## Error Log
```
# paste error here
```
```

## Feature Requests

We welcome feature suggestions! When proposing new features:

1. **Use case**: Describe the problem or use case
2. **Proposed solution**: How you envision the feature working
3. **Alternatives**: Any alternative approaches considered
4. **Implementation notes**: Technical details if you have ideas

### Feature Request Template

```markdown
## Feature Description
Brief description of the proposed feature

## Motivation
Why is this feature needed? What problem does it solve?

## Proposed Implementation
How should this feature work?

## Example Usage
```python
# Show example code using the feature
```

## Alternatives Considered
What other approaches did you consider?

## Additional Context
Any other information, diagrams, or references
```

## Areas for Contribution

We especially welcome contributions in these areas:

### High Priority

- 🧪 **Unit tests**: Increase test coverage
- 📚 **Documentation**: Improve docstrings and tutorials
- 🐛 **Bug fixes**: Address known issues
- 🎯 **Performance**: Optimize computational bottlenecks

### Medium Priority

- 🔬 **New experiments**: Add interesting experimental setups
- 📊 **Visualization**: Improve plotting and analysis tools
- 🤖 **Agent strategies**: Implement new selection strategies
- 🌐 **Environment types**: Add new hypothesis spaces

### Future Directions

- 🔄 **Active learning**: More sophisticated query strategies
- 🧠 **Deep learning**: Neural network-based agents
- 📈 **Scalability**: Handle larger hypothesis spaces
- 🎮 **Interactive demos**: Web-based visualization

## Questions?

If you have questions about contributing:

1. Check existing [issues](https://github.com/martinakaduc/teaching_simulation/issues)
2. Read the [README](README.md) and code documentation
3. Open a new issue with the `question` label

## Recognition

Contributors will be:

- Listed in the project's contributors page
- Acknowledged in release notes
- Credited in academic citations (where appropriate)

Thank you for helping improve TeachSim! 🎉
