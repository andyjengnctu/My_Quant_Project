"""Explicit build admission for isolated Oracle fixtures, never production IO.

Runtime gate tests use unittest.TestCase directly. Numerical/source mutation
tests retain all real contract and legacy-origin validators; this fixture seam
only avoids depending on a developer checkout's published current certificate.
"""
from contextlib import contextmanager
from unittest import TestCase
from unittest.mock import patch


@contextmanager
def oracle_fixture_build():
    from services.research.oracle_result_contract import oracle_component_revisions
    with patch("services.research.oracle_build_attestation.require_oracle_build_attestation",
               return_value={"component_revisions": oracle_component_revisions(),
                             "legacy_origins": {}, "fixture_only": True}):
        yield


class OracleFixtureTestCase(TestCase):
    def run(self, result=None):
        with oracle_fixture_build():
            return super().run(result)
