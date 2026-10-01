#!/usr/bin/env bash
set -euo pipefail

CRAN_MIRROR="${CRAN_MIRROR:-https://cloud.r-project.org}"
JAVA_HOME_DEFAULT="${JAVA_HOME_DEFAULT:-/usr/lib/jvm/default-java}"

if ! command -v sudo >/dev/null 2>&1; then
  echo "This installer needs sudo to install system packages." >&2
  exit 1
fi

sudo apt update -qq
sudo apt install -y --no-install-recommends \
  ca-certificates \
  dirmngr \
  gnupg \
  lsb-release \
  software-properties-common \
  wget

if [ "$(lsb_release -is)" != "Ubuntu" ]; then
  echo "This script follows the CRAN Ubuntu repository setup. Detected: $(lsb_release -is)." >&2
  exit 1
fi

wget -qO- "${CRAN_MIRROR}/bin/linux/ubuntu/marutter_pubkey.asc" \
  | sudo tee /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc >/dev/null

sudo add-apt-repository -y "deb ${CRAN_MIRROR}/bin/linux/ubuntu $(lsb_release -cs)-cran40/"

sudo apt update -qq
sudo apt install -y --no-install-recommends \
  cmake \
  default-jdk \
  g++ \
  gdal-bin \
  libcurl4-openssl-dev \
  libfontconfig1-dev \
  libfribidi-dev \
  libgdal-dev \
  libgeos-dev \
  libharfbuzz-dev \
  libhdf5-dev \
  libjpeg-dev \
  libpng-dev \
  libproj-dev \
  libssl-dev \
  libtiff5-dev \
  libxml2-dev \
  make \
  pkg-config \
  r-base \
  r-base-dev

export JAVA_HOME="${JAVA_HOME_DEFAULT}"
export PATH="${JAVA_HOME}/bin:${PATH}"

ensure_bashrc_line() {
  local line="$1"
  if [ -f "${HOME}/.bashrc" ] && grep -qxF "${line}" "${HOME}/.bashrc"; then
    return
  fi
  printf '\n%s\n' "${line}" >> "${HOME}/.bashrc"
}

ensure_bashrc_line "export JAVA_HOME=${JAVA_HOME_DEFAULT}"
ensure_bashrc_line 'export PATH=$JAVA_HOME/bin:$PATH'

sudo R CMD javareconf

sudo env CRAN_MIRROR="${CRAN_MIRROR}" Rscript -e '
options(repos = c(CRAN = Sys.getenv("CRAN_MIRROR", "https://cloud.r-project.org")))
packages <- c(
  "raster",
  "dismo",
  "pROC",
  "tidyverse",
  "rjson",
  "yaml",
  "rJava",
  "hdf5r",
  "arrow"
)
missing <- setdiff(packages, rownames(installed.packages()))
if (length(missing) > 0) {
  install.packages(missing)
} else {
  message("All requested R packages are already installed.")
}
'

echo "R environment setup complete."
echo "Open a new shell, or run: source ~/.bashrc"
