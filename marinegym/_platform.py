import os
from pathlib import Path


DEFAULT_EXPERIENCE = "omni.isaac.sim.python.kit"


def resolve_experience_path(
    environment: dict[str, str] | None = None,
    experience_name: str = DEFAULT_EXPERIENCE,
) -> str:
    """Resolve an Isaac Sim experience on both Linux and Windows installs."""
    env = os.environ if environment is None else environment
    search_roots = []

    if env.get("EXP_PATH"):
        search_roots.append(Path(env["EXP_PATH"]))
    if env.get("ISAACSIM_PATH"):
        search_roots.append(Path(env["ISAACSIM_PATH"]) / "apps")

    for root in search_roots:
        candidate = root / experience_name
        if candidate.is_file():
            return str(candidate)

    configured = ", ".join(str(root) for root in search_roots) or "none"
    raise RuntimeError(
        f"Unable to find Isaac Sim experience '{experience_name}'. "
        f"Searched: {configured}. Set EXP_PATH to the Isaac Sim apps directory "
        "or ISAACSIM_PATH to the Isaac Sim installation directory."
    )
