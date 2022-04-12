with import ./../our-nixpkgs.nix;

python38Packages.buildPythonPackage rec {
  pname = "tabnet";
  version = "0.1.6";

  src = python38Packages.fetchPypi {
    inherit pname version;
    sha256 = "101gy0f10lzjqqsppnq41abmhy1jvv3jpjv7a0jkax5nghch3fg7";
  };

  doCheck = false;
}
