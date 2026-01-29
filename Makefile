.PHONY: help install test lint format docker-build docker-run docker-stop clean

help:
	@echo "Brain Tumor MLOps - Available Commands"
	@echo "======================================"
	@echo "install       - Install dependencies"
	@echo "test          - Run all tests"
	@echo "test-unit     - Run unit tests only"
	@echo "test-int      - Run integration tests"
	@echo "lint          - Run linting checks"
	@echo "format        - Format code with black and isort"
	@echo "docker-build  - Build Docker image"
	@echo "docker-run    - Run with docker-compose"
	@echo "docker-stop   - Stop docker-compose services"
	@echo "validate      - Validate model before deployment"
	@echo "clean         - Clean up temporary files"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=app --cov-report=html --cov-report=term

test-unit:
	pytest tests/test_*.py -v

test-int:
	pytest tests/integration/ -v

lint:
	flake8 app/ tests/ --max-line-length=100
	pylint app/ --disable=C0111,R0903

format:
	black app/ tests/
	isort app/ tests/

docker-build:
	docker build -t brain-tumor-api:latest .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

docker-logs:
	docker-compose logs -f api

validate:
	python scripts/validate_model.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	rm -f .coverage
	rm -rf logs/*.log

dev:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

prod:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
