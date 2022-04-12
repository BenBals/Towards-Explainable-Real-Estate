with import ./../our-nixpkgs.nix;

python38Packages.buildPythonPackage rec {
  pname = "lsq-ellipse";
  version = "2.0.1";

  src = python38Packages.fetchPypi {
    inherit pname version;
    sha256 = "1lz67gnihddy0bg24vhv2pb47vslpk7h213yi70gqhxq4n5i4f0r";
  };

  propagatedBuildInputs = with python38Packages; [
    numpy
  ];

  doCheck = false;
}
