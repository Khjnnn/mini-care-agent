import pytest

from care_agent.fhir.client import FHIRClient
from care_agent.fhir.memory_server import InMemoryFHIR
from care_agent.tools.fhir_tools import build_fhir_tools


@pytest.fixture
def fhir_server() -> InMemoryFHIR:
    return InMemoryFHIR()


@pytest.fixture
async def fhir(fhir_server):
    async with FHIRClient("http://fhir.test/fhir", transport=fhir_server) as client:
        yield client


@pytest.fixture
def registry(fhir):
    return build_fhir_tools(fhir)
