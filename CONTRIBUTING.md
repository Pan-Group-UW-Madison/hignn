# Contributing Guide

Thanks for your interest in contributing to $\mathcal{H}$-HIGNN!

We are thrilled that you're interested in helping out. This document provides guidelines for contributing to the project, reporting issues, and seeking support. Following these guidelines helps make the process smooth and effective for everyone involved.

- [Code of Coduct](#code-of-conduct)

- [Ways to Contribute](#ways-to-contribute)

## Code of Conduct

By participating in this project, you are expected to uphold our Code of Conduct. We are committed to providing a friendly, safe, and welcoming environment for all, regardless of level of experience, gender, gender identity and expression, sexual orientation, disability, personal appearance, body size, race, ethnicity, age, religion, or nationality.

We do not tolerate harassment of participants in any form. Please be respectful and considerate of others.

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by contacting the project team at [Your Email] or via a private message to the maintainers. All complaints will be reviewed and investigated promptly and fairly.

## Ways to Contribute

We love contributions from our community! There are many ways to help, not just by writing code:

- 🐛 Reporting bugs with detailed information.

- 💡 Suggesting new features or enhancements.

- 📝 Improving documentation (fixing typos, clarifying explanations, adding examples).

- 🔍 Answering questions from other users on issues or discussions.

- 🧪 Writing or refining tests.

- 🌎 Translating the software or its documentation.

1. __Reporting Issues or Problems__

    Think you've found a bug or have a problem? Please let us know by creating a [GitHub Issue](https://docs.github.com/en/issues).

    Before you report an issue, please do a quick search to see if it has already been reported.

    How to Write a Good Bug Report
    To help us resolve your issue quickly, please include as much of the following as possible:

    - A clear, descriptive title.

    - A detailed description of the problem.

    - Steps to Reproduce: The exact steps to recreate the issue.

    - Expected Behavior: What you expected to happen.

    - Actual Behavior: What actually happened.

    - Screenshots/Logs: If applicable, add screenshots or copy-paste any error logs.

    - Your Environment:

        OS: [e.g., Ubuntu 22.04]

        GPU driver version: [e.g., cuda-12.8]

2. __Seeking Support__

    We want to help you succeed! Here's how to get support:

    - Check the Documentation: First, please check our [Readme.md](Readme.md) for answers.

    - Search Existing Issues: Check the [GitHub Issues](https://github.com/Pan-Group-UW-Madison/hignn/issues) to see if your question has already been asked and answered.

    - Open a Support Issue: If you can't find an answer, feel free to open a new issue and label it as a question or support.

    Please be patient. This project is maintained by a very small research group (1~2 PhD students and one professor) at University of Wisconsin-Madison. We will do our best to respond as quickly as possible.

3. __Contributing to the Software__

    We welcome all code contributions! To ensure a smooth process, please follow these steps.

    **Proposing a New Feature**

    - Open an Issue First: Before you write a lot of code, please open an issue to discuss the new feature.

    - Explain the Why: Describe the problem you're solving or the value the feature adds.

    - Discuss the Approach: Outline how you plan to implement it. This allows the maintainers to provide feedback and guidance early on.

    **The Pull Request Process**

    - Fork the Repository: Click the "Fork" button on Github and clone your fork locally.

    - Create a Branch: Create a new branch for your feature or bugfix.

        ```[bash]
        git checkout -b feature/your-amazing-feature
        # or
        git checkout -b fix/annoing-bug
        ```
    
    - Make Your Changes: Write your code and tests

        - Follow the exiting code style and conventions.

        - Ensure all tests pass. (Add tests for new functionality!)

        - Update the documentation if necessary.

    - Commit Your Changes: Write clear, descriptive commit messages.

        ```[bash]
        git commit -m "feat: detailed feature"
        ```
    
    - Push to Your Forks:

        ```[bash]
        git push origin feature/your-amazing-feature
        # or
        git push origin fix/annoying-bug
        ```
    
    - Submit a Pull Request (PR): Go to the original repository and open a Pull Request.

        - Link the Issue: In the PR description, mention the issue it fixes (e.g., Fixes #45).

        - Describe Your Changes: Explain what you did and why.

        - Wait for Review: A maintainer will review your PR. They may suggest changes. This is a normal part of the process!

    Once your PR is approved and all checks pass, a maintainer will merge it. Congratulations, you're now a contributor!

Thank you for being a part of our community!