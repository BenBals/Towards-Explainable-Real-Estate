let
  pkgs = import ./our-nixpkgs.nix;
  lsq-ellipse = import ./nix/lsq-ellipse.nix;
  alphashape = import ./nix/alphashape.nix;
  pgeocode = import ./nix/pgeocode.nix;
  tabnet = import ./nix/tabnet.nix;
in
  (with pkgs; [
    # things needed to run jupyter
    python38
    jupyter

    # custom build python packages
    lsq-ellipse
    alphashape
    pgeocode
    tabnet
  ]) ++ (with pkgs.python38Packages; [
    jupyter
    jupyter_core
    jupyterlab

    # for tensorflow
    pip
    setuptools

    # dependencies for our scripts and notebooks
    pymongo
    pandas
    geopandas
    Rtree
    numpy
    matplotlib
    statsmodels
    scikitlearn
    sklearn-deap
    faker
    flake8
    pytest
    pylint
    deap
    tqdm
  ])
