"""The dependency audit captures provenance metadata but makes no rights claims."""
import pytest
from scripts.dependency_inventory import build_inventory, outside_repo, ROOT


def test_dependency_manifest_marks_metadata_unverified():
    data = build_inventory()
    assert data["schema_version"] == 1
    assert len(data["pyproject_sha256"]) == 64
    assert "numpy" in {p["name"].lower() for p in data["packages"]}
    assert data["warning"].startswith("Package metadata")


def test_inventory_requires_private_destination():
    with pytest.raises(ValueError):
        outside_repo(ROOT / "private" / "package_inventory.json")
