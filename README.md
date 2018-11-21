# d3m_croc
Character recognition and object classification system for images

## Quick Start

Use CROC in your project via pip with `pip3 install -e <path/to/croc>`.

or

Start CROC as a service on your local-machine with:

1) `docker build -t croc-http:dev -f ./http.dockerfile .`
2) `docker run -p 5000:5000 croc-http:dev`

## Structure of this repo

The core of this repo is `setup.py` and `d3m_croc`. 

This repo is pip-installsable and makes the contents of `d3m_croc` available after installation.

There is a flask wrapper for the library located in `http-wrapper`. It uses `nk_croc` and can be built with the `http.dockerfile`. For more information see [the README.md in `http-wrapper`](./http-wrapper/README.md)

## Coming soon

- Other wrappers for croc that are segmented out in other repos?
