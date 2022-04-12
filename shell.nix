with import ./our-nixpkgs.nix;

let
  localPath = ./local.nix;
  localInputs = lib.optional (builtins.pathExists localPath) (import localPath).inputs;
  localHooks = lib.optionalString (builtins.pathExists localPath) (import localPath).hooks;
in mkShell {
  buildInputs = import ./dependencies.nix ++ localInputs;

  shellHook = ''
    # link libcuda.so.1 without stuff we dont need
    # mkdir -p _libcuda
    # ln -f -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 _libcuda/libcuda.so.1
    export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:/nix/store/4kswv5jhha9hjnwkw84900pzhgb9dalj-cudatoolkit-10.1.243-lib/lib/:/usr/local/cuda-10.2/targets/x86_64-linux/lib/l:$LD_LIBRARY_PATH

    export PYTHONPATH="$(pwd)/python-db:$(pwd)/_build/lib/python3.8/site-packages:$PYTHONPATH"
    export PYTHONUSERBASE="$(pwd)/_build"
    export PATH="$(pwd)/_build/bin/:$PATH"
    unset SOURCE_DATE_EPOCH
  '' + localHooks;
}
