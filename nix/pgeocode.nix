with import ./../our-nixpkgs.nix;

python38Packages.buildPythonPackage rec {
  pname = "pgeocode";
  version = "0.3.0";

  src = python38Packages.fetchPypi {
    inherit pname version;
    sha256 = "19lvv3y2cilnaxs0nr3jzcjvjcd5s2473h382vbfw00k8dix3108";
  };

  propagatedBuildInputs = with python38Packages; [
    pandas
    requests
  ];

  doCheck = false;
}
