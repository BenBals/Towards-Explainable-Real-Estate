with import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/1e73714c9ab56894017469775c4413d91e3a357f.tar.gz") {};

let
  my-r-pkgs = rWrapper.override {
    packages = with rPackages; [
      MLmetrics
      broom
      cowplot
      dplyr
      ggplot2
      ggthemes
      glue
      gridExtra
      mongolite
      rgdal
      rgeos
      stringr
      tidyr
      tidyverse
      caret
    ];
  };
in mkShell {
  buildInputs = with pkgs;[
    git
    glibcLocales
    openssl
    which
    openssh
    curl
    wget

    rstudio
    my-r-pkgs
  ];
  inputsFrom = [ my-r-pkgs ];
  shellHook = ''
    mkdir -p "$(pwd)/_libs"
    export R_LIBS_USER="$(pwd)/_libs"
  '';
  GIT_SSL_CAINFO = "${cacert}/etc/ssl/certs/ca-bundle.crt";
  LOCALE_ARCHIVE = "${glibcLocales}/lib/locale/locale-archive";
}
