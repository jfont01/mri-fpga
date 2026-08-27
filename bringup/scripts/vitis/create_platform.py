import glob
import os
import shutil
from pathlib import Path
from typing import List, Optional

import vitis


def env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(
            "required environment variable {} is missing".format(name)
        )
    return value


def find_one(root: Path, patterns: List[str]) -> Optional[Path]:
    candidates = []  # type: List[Path]

    for pattern in patterns:
        matches = glob.glob(
            str(root / "**" / pattern),
            recursive=True,
        )

        candidates.extend(Path(p) for p in matches)

    candidates = [
        p
        for p in candidates
        if p.is_file()
    ]

    if not candidates:
        return None

    # Prefer exported/generated artifacts over files buried
    # deeper inside source/generated trees.
    candidates.sort(
        key=lambda p: (
            "export" not in p.parts,
            len(p.parts),
            str(p),
        )
    )

    return candidates[0]


# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------

workspace = Path(env("VITIS_WORKSPACE")).resolve()
platform_name = env("PLATFORM_NAME")
xsa = Path(env("XSA")).resolve()
xpfm_out = Path(env("XPFM_OUT")).resolve()
results_dir = Path(env("RESULTS_DIR")).resolve()


# -----------------------------------------------------------------------------
# Validate inputs
# -----------------------------------------------------------------------------

if not xsa.is_file():
    raise RuntimeError(
        "XSA not found: {}".format(xsa)
    )


# -----------------------------------------------------------------------------
# Prepare deterministic workspace
#
# A dedicated workspace means recreation is deterministic and does not risk
# deleting an unrelated Vitis project.
# -----------------------------------------------------------------------------

shutil.rmtree(
    str(workspace),
    ignore_errors=True,
)

workspace.mkdir(
    parents=True,
    exist_ok=True,
)

results_dir.mkdir(
    parents=True,
    exist_ok=True,
)

(results_dir / "boot").mkdir(
    parents=True,
    exist_ok=True,
)


# -----------------------------------------------------------------------------
# Create platform
# -----------------------------------------------------------------------------

print(
    "Creating bare-metal Vitis platform '{}'".format(
        platform_name
    )
)

print(
    "  workspace: {}".format(
        workspace
    )
)

print(
    "  XSA      : {}".format(
        xsa
    )
)


client = vitis.create_client()

client.set_workspace(
    path=str(workspace)
)

platform = client.create_platform_component(
    name=platform_name,
    hw_design=str(xsa),
    os="standalone",
    cpu="psu_cortexa53_0",
    domain_name="standalone_psu_cortexa53_0",
)

print("")
print("Platform domains before build:")

for domain in platform.list_domains():
    print(
        "  {}  cpu={}  os={}".format(
            domain.get("domain_name"),
            domain.get("processor"),
            domain.get("os"),
        )
    )

platform.build()


# -----------------------------------------------------------------------------
# Locate generated XPFM
# -----------------------------------------------------------------------------

expected = (
    workspace
    / platform_name
    / "export"
    / platform_name
    / "{}.xpfm".format(platform_name)
)

if not expected.is_file():

    found = find_one(
        workspace,
        [
            "{}.xpfm".format(platform_name),
            "*.xpfm",
        ],
    )

    if found is None:
        raise RuntimeError(
            "platform build completed but no .xpfm was found"
        )

    expected = found


# -----------------------------------------------------------------------------
# Export XPFM
# -----------------------------------------------------------------------------

xpfm_out.parent.mkdir(
    parents=True,
    exist_ok=True,
)

shutil.copy2(
    str(expected),
    str(xpfm_out),
)

print(
    "XPFM exported: {} <- {}".format(
        xpfm_out,
        expected,
    )
)


# -----------------------------------------------------------------------------
# Export boot/debug artifacts when Vitis generated them
#
# Their internal paths are Vitis-generated, so discover them instead of
# assuming a GUI/workspace directory layout.
# -----------------------------------------------------------------------------

artifacts = {
    "psu_init.tcl": [
        "psu_init.tcl",
    ],

    "boot/fsbl.elf": [
        "zynqmp_fsbl.elf",
        "fsbl.elf",
        "*fsbl*.elf",
    ],

    "boot/pmufw.elf": [
        "zynqmp_pmufw.elf",
        "pmufw.elf",
        "*pmufw*.elf",
    ],
}


for rel_out, patterns in artifacts.items():

    src = find_one(
        workspace,
        patterns,
    )

    dst = results_dir / rel_out

    if src is None:
        print(
            "WARNING: platform artifact not found: {}".format(
                rel_out
            )
        )
        continue

    dst.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        str(src),
        str(dst),
    )

    print(
        "Exported: {} <- {}".format(
            dst,
            src,
        )
    )