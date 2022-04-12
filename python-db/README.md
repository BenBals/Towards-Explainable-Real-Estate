# python-db
It is recommended to install requirements in a virtual environment:

``` sh
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

To exit the virtual environment simply run
``` sh
deactivate
```

To add additional requirements, please use [pipreqs](https://github.com/bndr/pipreqs).

## Alternative
You can also use `nix-shell` to achive the same result. Simply run `nix-shell` in the root of the repo to get a shell that has all dependencies installed. Exit the shell like normal.

This is especially recommended for running the scripts on the server as it only requires the manual installation of nix and no other tools.
