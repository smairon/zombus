"""
Pytest configuration for pancho tests.
"""

from typing import Any

import pytest
from zodchy.toolbox.di import DIResolverContract

from zombus.registration.registry import ActorsRegistry


# Simple real implementations for testing
class SimpleDIResolver:
    """Simple implementation of DIResolverContract for testing."""

    def __init__(self, dependencies: dict[type, Any] | None = None):
        self._dependencies = dependencies or {}

    async def resolve(self, contract: type, context: Any = None) -> Any:
        """Resolve a dependency by type."""
        if contract in self._dependencies:
            return self._dependencies[contract]
        # Try to return a default instance if not found, but handle abstract classes
        try:
            return contract()
        except (TypeError, ValueError):
            # If we can't instantiate (e.g., abstract class), return None
            # This matches the behavior when dependency is not registered
            return None

    async def shutdown(self, context: Any = None) -> None:
        """Shutdown the resolver."""
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


class SimpleDIContainer:
    """Simple implementation of DIContainerContract for testing."""

    def __init__(self, dependencies: dict[type, Any] | None = None):
        self._dependencies = dependencies or {}
        self._resolvers_created = 0

    def register_dependency(
        self,
        implementation: Any,
        contract: type | None = None,
        cache_scope: str | None = None,
    ) -> None:
        """Register a dependency."""
        contract_type = contract or type(implementation)
        self._dependencies[contract_type] = implementation

    def register_callback(
        self,
        contract: type,
        callback: Any,
        trigger: str,
    ) -> None:
        """Register a callback."""
        pass

    def get_resolver(self, *context: Any) -> DIResolverContract:
        """Get a dependency resolver."""
        self._resolvers_created += 1
        return SimpleDIResolver(self._dependencies)

    async def shutdown(self, context: Any = None) -> None:
        """Shutdown the container."""
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


# Ensure pytest-asyncio plugin is loaded
pytest_plugins = ("pytest_asyncio",)


@pytest.fixture
def actors_registry() -> ActorsRegistry:
    """Fixture for ActorsRegistry."""
    return ActorsRegistry()


@pytest.fixture
def dependency_resolver() -> SimpleDIResolver:
    """Fixture for SimpleDIResolver."""
    return SimpleDIResolver()


@pytest.fixture
def dependency_resolver_with_deps() -> SimpleDIResolver:
    """Fixture for SimpleDIResolver with test dependencies."""
    # Import here to avoid circular import at module level
    # This is safe because conftest is loaded before test files
    try:
        from tests.test_processing import TestDependency
    except ImportError:
        # Fallback if import fails (shouldn't happen in normal test runs)
        class TestDependency:
            """Test dependency class for testing."""

            pass

    test_dep = TestDependency()
    return SimpleDIResolver({TestDependency: test_dep})


@pytest.fixture
def dependency_container() -> SimpleDIContainer:
    """Fixture for SimpleDIContainer."""
    return SimpleDIContainer()
