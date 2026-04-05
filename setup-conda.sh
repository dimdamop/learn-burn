#!/bin/bash


env_path='./conda_env'

# We use self-contained environments, never global configs
mv ~/.condarc ~/dot_condarc >/dev/null 2>&1 || true

unset PYTHONPATH

arch=$(uname -m)   # x86_64 or aarch64
fname="Miniforge3-Linux-${arch}.sh"

if [ ! -f $fname ]
then
    # install miniconda
    wget \
        https://github.com/conda-forge/miniforge/releases/latest/download/$fname \
        -O $fname
fi

if [ ! -d miniconda3 ]
then
    bash $fname -b -p miniconda3
    # activate miniconda
    chmod u+x miniconda3/bin/*
    source miniconda3/etc/profile.d/conda.sh
    conda config --set ssl_verify false
    conda config --set auto_activate_base false
fi

if [ ! -d "$env_path" ]
then
    conda env create -f conda-init.yaml --prefix="$env_path"
    conda activate "$env_path"
    pip install torch

    # Set libtorch env vars on every activation.
    activate_dir="$env_path/etc/conda/activate.d"
    deactivate_dir="$env_path/etc/conda/deactivate.d"
    mkdir -p "$activate_dir" "$deactivate_dir"

    cat > "$activate_dir/libtorch.sh" << 'EOF'
#!/bin/bash
_torch_dir="$CONDA_PREFIX/lib/python3.12/site-packages/torch"
export LIBTORCH="$_torch_dir"
export LIBTORCH_CXX11_ABI=1
export _OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$_torch_dir/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
EOF

    cat > "$deactivate_dir/libtorch.sh" << 'EOF'
#!/bin/bash
unset LIBTORCH
unset LIBTORCH_CXX11_ABI
export LD_LIBRARY_PATH="$_OLD_LD_LIBRARY_PATH"
unset _OLD_LD_LIBRARY_PATH
EOF
else
    source miniconda3/etc/profile.d/conda.sh
    conda activate "$env_path"
fi
