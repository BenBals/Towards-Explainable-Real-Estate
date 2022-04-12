with import ./../our-nixpkgs.nix;

python38Packages.buildPythonPackage rec {
  pname = "alphashape";
  version = "1.1.0";

  src = python38Packages.fetchPypi {
    inherit pname version;
    sha256 = "0kmr7glxk4y69a0ah4pwi8hja7s46n2x688m0wz43ssfgi5vzlik";
  };

  propagatedBuildInputs = with python38Packages; [
    click
    click-log
    shapely
    scipy
  ];

  doCheck = false;
}
