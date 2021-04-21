install:
	pip install -e .

dev-install:
	pip install -r dev_requirements.txt
	pip install -e .

unit-test:
	coverage run --branch --source=bcipy -m pytest -m "not slow"
	coverage report

integration-test:
	coverage run --branch --source=bcipy -m pytest -m "slow"
	coverage report

test-all:
	coverage run --branch --source=bcipy -m pytest
	coverage report

coverage-html:
	coverage run --branch --source=bcipy -m pytest
	coverage html

format:
	python -m isort --profile black bcipy
	python -m black --line-length 120 bcipy

lint:
	python -m mypy --ignore-missing-imports bcipy
	python -m flake8 --ignore=E203 --max-line-length 120 --statistics --show-source bcipy

clean:
	find . -name "*.py[co]" -o -name __pycache__ -exec rm -rf {} +
	find . -path "*/*.pyo"  -delete
	find . -path "*/*.pyc"  -delete

bci-gui:
	python bcipy/gui/BCInterface.py
