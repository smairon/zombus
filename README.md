# Zodchy

## Установка uv (если не установлен)
curl -LsSf https://astral.sh/uv/install.sh | sh

## Установка проекта и dev-зависимостей
uv sync --all-extras

## Активация виртуального окружения
source .venv/bin/activate  # Linux/Mac
## или
.venv\Scripts\activate     # Windows

## Запуск тестов
uv run pytest

## Запуск линтеров
uv run black src tests
uv run ruff check --fix src tests
uv run mypy src

## Проверка безопасности
uv run safety check
uv run bandit -r src

## Запуск pre-commit на всех файлах
uv run pre-commit run --all-files

## Сборка пакета
uv build

## Публикация пакета
uv publish
