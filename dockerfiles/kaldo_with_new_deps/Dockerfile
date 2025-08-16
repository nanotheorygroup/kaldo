# -----------------------------------------------------------------------------
# Dockerfile: Build + Test + Docs for kALDo (using YOUR requirements.in)
# -----------------------------------------------------------------------------
# Pipeline (kept minimal, with strong comments):
#   1) Pin/install EXACT Python deps from your requirements.in via pip-compile
#   2) Clone kALDo at a selectable ref and install it *editable* (no deps)
#      - Editable keeps imports pointing at this checkout for pytest & autodoc
#      - --no-deps preserves YOUR pinned dependency set
#   3) Run pytest from the repository ROOT (tests use relative data paths)
#   4) Build Sphinx docs once using the repo’s Makefile target
#      - Stage HTML into /opt/kaldo-docs (a clean, stable path)
#      - Hard fail if index.html is missing (prevents 404s later)
#   5) Serve the built docs on container port 8000
#
# -----------------------------------------------------------------------------

FROM continuumio/miniconda3

# -----------------------------------------------------------------------------
# Environment hygiene:
# - DEBIAN_FRONTEND=noninteractive avoids interactive prompts in apt
# - PYTHONUNBUFFERED=1 for immediate, unbuffered logs (pytest / sphinx errors)
# - TZ/LANG ensure predictable timestamps and UTF-8 (Sphinx can be picky)
# -----------------------------------------------------------------------------
ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 TZ=Etc/UTC LANG=en_US.utf8

# -----------------------------------------------------------------------------
# System packages (lean):
# - locales: generate en_US.UTF-8 (Sphinx/Unicode safety)
# - git: clone the kALDo repo
# - make: needed because we call `make -C docs html`
#   (No compilers/graphviz/etc. unless your docs actually require them.)
# -----------------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends locales git make && \
    localedef -i en_US -c -f UTF-8 \
              -A /usr/share/locale/locale.alias en_US.UTF-8 && \
    rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Python bootstrap:
# - Use pip + wheel + pip-tools; YOUR requirements.in drives everything
# - We intentionally do not `conda install ...` to keep behavior consistent
#   with your pip-compile workflow.
# -----------------------------------------------------------------------------
RUN python -m pip install --upgrade pip setuptools wheel pip-tools

# -----------------------------------------------------------------------------
# Dependency pinning (UNCHANGED FLOW):
# - Copy only requirements.in to maximize Docker layer cache reuse
# - pip-compile -> requirements.txt (locks versions)
# - Install exactly those pins (no surprise upgrades)
# -----------------------------------------------------------------------------
WORKDIR /mnt/work
COPY requirements.in ./
RUN pip-compile requirements.in --output-file=requirements.txt --no-strip-extras && \
    python -m pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Fetch kALDo source:
# - KALDO_VERSION is a build-arg so you can pick branch/tag/commit
# - We clone into /tmp/kaldo (a conventional build workspace)
#   (If the repo uses Git LFS for test assets, consider adding `git-lfs` and
#    `git lfs pull` here. We’ve kept it minimal since you asked for no extras.)
# -----------------------------------------------------------------------------
ARG KALDO_VERSION=main
WORKDIR /tmp/
RUN git clone --branch "${KALDO_VERSION}" \
      https://github.com/nanotheorygroup/kaldo.git /tmp/kaldo

# -----------------------------------------------------------------------------
# Install kALDo EDITABLE:
# - Critical for pytest and Sphinx autodoc to import THIS checkout
# - --no-deps preserves YOUR pinned env (prevents setup.py install_requires
#   from pulling conflicting versions)
# - If you comment this out, expect import/cwd path issues in tests/docs.
# -----------------------------------------------------------------------------
#RUN python -m pip install --no-cache-dir --no-deps -e /tmp/kaldo

# -----------------------------------------------------------------------------
# Run tests:
# - WORKDIR is set to the repo root so tests that open relative paths like
#   'kaldo/tests/si-amorphous/CONFIG' resolve correctly.
# - We target the package tests explicitly to avoid picking up doctests
#   or misc files outside that tree.
# - --maxfail=1 for fail-fast; -q for clean logs; --disable-warnings to keep
#   output focused (tune as you like).
# -----------------------------------------------------------------------------
WORKDIR /tmp/kaldo
RUN python -m pytest kaldo/tests --maxfail=1 --disable-warnings -q

# -----------------------------------------------------------------------------
# Build docs once and stage them:
# - We use the repo’s Makefile target: `make -C docs html`
# - SPHINXOPTS:
#     -j auto      => parallel build (fast)
#     --keep-going => continue after first warning to report all (but DO NOT
#                     hide true errors; make will still fail non-zero on errors)
# - We normalize the output into /opt/kaldo-docs regardless of where the
#   Makefile placed HTML (commonly docs/_build/html).
# - The `test -f index.html` assertion aborts the build if docs didn’t render.
# -----------------------------------------------------------------------------
RUN make -C docs html SPHINXOPTS="-j auto --keep-going" \
 && mkdir -p /opt/kaldo-docs \
 && cp -a docs/_build/html/. /opt/kaldo-docs/ \
 && test -f /opt/kaldo-docs/index.html

# -----------------------------------------------------------------------------
# Serve the built docs:
# - EXPOSE 8000 is documentation-only (doesn’t open the port by itself)
# - CMD runs a tiny Python HTTP server that serves the STATIC HTML we staged
#   at /opt/kaldo-docs.
# - On the host, run: `docker run --rm -p 8000:8000 <tag>` and open
#   http://localhost:8000
# -----------------------------------------------------------------------------
EXPOSE 8000
CMD ["python","-m","http.server","8000","--directory","/opt/kaldo-docs"]

