# Project Coding Standards

## Python Guidelines
- Use Python 3.x for all code.
- Adhere to PEP 8 style guide for code formatting.
- Follow functional programming principles where beneficial.
- Utilize type hints for function and variable annotations.
- Prefer immutable data structures when appropriate.
- Employ list comprehensions and generator expressions for conciseness.
- Maintain a logical project structure with clear modules and packages.
- Write clear docstrings for modules, classes, functions, and methods (e.g., NumPy or Google style).
- Comment complex logic and non-obvious parts above the code. Avoid side comments.

## Flask Guidelines
- Structure your application using blueprints for modularity.
- Utilize Flask's extensions for common functionalities (e.g., Flask-SQLAlchemy, Flask-Migrate, Flask-WTF).
- Follow best practices for defining routes and view functions.
- Keep view functions focused on handling requests and responses. Move business logic to separate modules or services.
- Utilize Flask's template engine (Jinja2) effectively and keep template logic minimal.
- Follow conventions for form handling with Flask-WTF.

## Build Tool and Environment
- Specify dependencies in `requirements.txt`.
- Use a virtual environment (e.g., `venv`).
- Manage environment variables using `.env` files.

## Naming Conventions
- Use `snake_case` for variables, functions, and modules.
- Use `CamelCase` for classes.
- Use `UPPER_SNAKE_CASE` for constants.
- Prefix private class members with `_`.

## Error Handling
- Use `try...except` blocks for potential exceptions, catching specific exceptions.
- Implement custom exception classes for better context.
- Log errors using Python's `logging` module with relevant information.
- Use context managers (`with`) for resource management.