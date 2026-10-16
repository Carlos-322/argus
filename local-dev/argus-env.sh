# Source this file from bash to enter the Argus development environment.
_argus_project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" || return 1

export UV_PYTHON_INSTALL_DIR=/mnt/d/applications/uv/python-linux
export UV_CACHE_DIR=/mnt/d/applications/uv/cache-linux
export UV_PROJECT_ENVIRONMENT=/mnt/d/applications/uv/environments/argus-linux
export UV_LINK_MODE=copy
export PATH="/mnt/d/applications/uv/linux:/mnt/d/applications/uv/tools/ffmpeg-linux/bin:$PATH"

if [[ ! -f "$UV_PROJECT_ENVIRONMENT/bin/activate" ]]; then
    printf 'Argus Linux environment not found: %s\n' "$UV_PROJECT_ENVIRONMENT" >&2
    unset _argus_project_root
    return 1
fi

source "$UV_PROJECT_ENVIRONMENT/bin/activate"
cd -- "$_argus_project_root" || return 1
unset _argus_project_root

if [[ $- == *i* ]]; then
    printf '\nArgus environment ready (WSL Ubuntu 22.04 / Python 3.11.15).\n'
    printf 'Try: uv run --frozen python -m review --help\n'
    printf 'Windows D:\\ paths are available under /mnt/d/\n\n'
fi
