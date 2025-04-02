#!/usr/bin/env bash

# ## strict bash mode
set -eEuo pipefail

# ## run via
# ##    . load_env.sh
# ## otherwise a subshell is used and the PATH / variable adjustments are only made there.
# module reset
module purge
module load DefaultModules

# ## install micromamba
if [[ ! -f /cfs/earth/scratch/${USER}/bin/micromamba ]]; then
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
    mkdir -p /cfs/earth/scratch/${USER}/bin
    mv bin/micromamba /cfs/earth/scratch/${USER}/bin
    rmdir bin
fi
# ## init micromamba
export MAMBA_ROOT_PREFIX="/cfs/earth/scratch/${USER}/.conda/"
eval "$("/cfs/earth/scratch/${USER}/bin/micromamba" shell hook -s posix)"


# ## get name of environment as specified in environment.yml
environment_file=environment.sa2.yml
env_name=$(sed -ne 's/^name: \(.*\)$/\1/p' ${environment_file:?})
echo "################## Load (and set up) environment ${env_name:?}"

# ## install env if it does not exist
if ! micromamba env list | grep -Eq "^\s*${env_name:?} "; then
    echo "- Create conda environment ${env_name:?}"

    micromamba -y create -f ${environment_file:?} || { echo "Environment creation failed!"; exit 1; }

else
    echo "- Environment ${env_name:?} already exists"

fi
