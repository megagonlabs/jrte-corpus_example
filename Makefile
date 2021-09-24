all: lint
lint: markdownlint pylint
pylint: flake8 black mypy isort pydocstyle yamllint

markdownlint:
	npx markdownlint '*.md'

TARGET_DIRS:=train.py

flake8:
	find *.py | xargs flake8
black:
	find *.py | xargs black --diff | diff /dev/null -
pyright:
	pyright
isort:
	find *.py | xargs isort --diff | diff /dev/null -
pydocstyle:
	pydocstyle *.py --ignore=D100,D101,D102,D103,D104,D105,D107,D203,D212

yamllint:
	find .github -name '*.yml' -type f | xargs yamllint --no-warnings

.PHONY: all flake8 black mypy isort pydocstyle yamllint
.DELETE_ON_ERROR:
