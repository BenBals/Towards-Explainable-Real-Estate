let
    pkgs = import ./our-nixpkgs.nix;
in
 pkgs.mkShell {
  buildInputs = with pkgs; [
    cargo
    clippy
    dhall
    openssl
    pkg-config
    rustc
    rustup
    rustfmt
    rust-analyzer
    rls
    valgrind
  ];
 }
