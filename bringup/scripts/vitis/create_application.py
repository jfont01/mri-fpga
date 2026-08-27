import glob
import os
import shutil
from pathlib import Path

import vitis


def env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(
            "required environment variable {} is missing".format(name)
        )
    return value


# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------

workspace = Path(env("VITIS_WORKSPACE")).resolve()
app_name = env("APP_NAME")

xpfm = Path(env("XPFM")).resolve()
platform_repo = Path(env("PLATFORM_REPO")).resolve()

# The platform component created by create_platform.py is named after
# the exported XPFM:
#
#   v0_platform.xpfm -> v0_platform
#
platform_name = xpfm.stem

domain_name = env("DOMAIN_NAME")
firmware_dir = Path(env("FIRMWARE_DIR")).resolve()
elf_out = Path(env("ELF_OUT")).resolve()


# -----------------------------------------------------------------------------
# Validate inputs
# -----------------------------------------------------------------------------

if not xpfm.is_file():
    raise RuntimeError(
        "XPFM not found: {}".format(xpfm)
    )

if not platform_repo.is_dir():
    raise RuntimeError(
        "platform repository not found: {}".format(platform_repo)
    )

if not firmware_dir.is_dir():
    raise RuntimeError(
        "firmware directory not found: {}".format(firmware_dir)
    )


# -----------------------------------------------------------------------------
# Discover firmware sources
# -----------------------------------------------------------------------------

source_exts = {
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".S",
    ".s",
}

sources = sorted(
    p
    for p in firmware_dir.rglob("*")
    if p.is_file()
    and p.suffix in source_exts
)

if not sources:
    raise RuntimeError(
        "no C/C++/assembly sources found under {}".format(
            firmware_dir
        )
    )


# -----------------------------------------------------------------------------
# Prepare deterministic application workspace
# -----------------------------------------------------------------------------

shutil.rmtree(
    str(workspace),
    ignore_errors=True,
)

workspace.mkdir(
    parents=True,
    exist_ok=True,
)

elf_out.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# -----------------------------------------------------------------------------
# Create Vitis client
# -----------------------------------------------------------------------------

print(
    "Creating Vitis application '{}'".format(
        app_name
    )
)

print(
    "  requested XPFM : {}".format(
        xpfm
    )
)

print(
    "  platform name  : {}".format(
        platform_name
    )
)

print(
    "  platform repo  : {}".format(
        platform_repo
    )
)

print(
    "  domain         : {}".format(
        domain_name
    )
)

print(
    "  firmware       : {}".format(
        firmware_dir
    )
)


client = vitis.create_client()

client.set_workspace(
    path=str(workspace)
)


# -----------------------------------------------------------------------------
# Register platform repository
# -----------------------------------------------------------------------------

print("")
print("Registering Vitis platform repository:")
print("  {}".format(platform_repo))

client.add_platform_repos(
    platform=str(platform_repo)
)


# -----------------------------------------------------------------------------
# Resolve platform using Vitis itself
#
# Do not immediately pass our manually constructed XPFM path to
# create_app_component().
#
# Vitis maintains its own platform repository database.  Asking Vitis to
# resolve the platform here proves that the platform is actually visible
# before we attempt to create the application.
# -----------------------------------------------------------------------------

print("")
print(
    "Resolving platform '{}' through Vitis repository...".format(
        platform_name
    )
)

resolved_xpfm = client.find_platform_in_repos(
    platform_name
)

print(
    "Vitis repository result: {!r}".format(
        resolved_xpfm
    )
)


if not resolved_xpfm:
    raise RuntimeError(
        "\n"
        "Vitis could not resolve platform '{}'.\n"
        "\n"
        "Platform repository:\n"
        "  {}\n"
        "\n"
        "Expected XPFM:\n"
        "  {}\n"
        "\n"
        "The XPFM exists on disk, but Vitis does not currently recognize\n"
        "it as a platform in its repository.\n".format(
            platform_name,
            platform_repo,
            xpfm,
        )
    )


resolved_xpfm = Path(
    str(resolved_xpfm)
).resolve()


print(
    "Resolved platform:"
)

print(
    "  {}".format(
        resolved_xpfm
    )
)


if not resolved_xpfm.is_file():
    raise RuntimeError(
        "Vitis resolved platform to a nonexistent XPFM: {}".format(
            resolved_xpfm
        )
    )


# -----------------------------------------------------------------------------
# Create application
# -----------------------------------------------------------------------------
print("")
print("Available embedded application templates:")

try:
    templates = client.get_templates(
        type="EMBD_APP"
    )

    for template in templates:
        print("  {}".format(template))

except Exception as exc:
    print(
        "WARNING: could not list embedded templates: {}".format(
            exc
        )
    )
    
print("")
print("Creating application component...")

app = client.create_app_component(
    name=app_name,
    platform=str(resolved_xpfm),
    domain=domain_name,
    template="empty_application",
)


# -----------------------------------------------------------------------------
# Import firmware
# -----------------------------------------------------------------------------

for src in sources:

    rel_parent = src.parent.relative_to(
        firmware_dir
    )

    if str(rel_parent) == ".":
        dest = "src"
    else:
        dest = "src/{}".format(
            rel_parent.as_posix()
        )

    app.import_files(
        from_loc=str(src.parent),
        files=[src.name],
        dest_dir_in_cmp=dest,
    )

    print(
        "  import {}".format(
            src.relative_to(firmware_dir)
        )
    )


# -----------------------------------------------------------------------------
# Build application
# -----------------------------------------------------------------------------

print("")
print("Building application...")

app.build()


# -----------------------------------------------------------------------------
# Locate generated ELF
# -----------------------------------------------------------------------------

pattern = str(
    workspace
    / "**"
    / "{}.elf".format(app_name)
)

candidates = [
    Path(p)
    for p in glob.glob(
        pattern,
        recursive=True,
    )
]

candidates = [
    p
    for p in candidates
    if p.is_file()
]

if not candidates:
    raise RuntimeError(
        "application build completed but {}.elf was not found".format(
            app_name
        )
    )


candidates.sort(
    key=lambda p: (
        "build" not in p.parts,
        len(p.parts),
        str(p),
    )
)

elf = candidates[0]


# -----------------------------------------------------------------------------
# Export ELF
# -----------------------------------------------------------------------------

shutil.copy2(
    str(elf),
    str(elf_out),
)

print(
    "ELF exported: {} <- {}".format(
        elf_out,
        elf,
    )
)