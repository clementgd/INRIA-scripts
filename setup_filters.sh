#!/bin/bash

nb_convert_path=$(find ~ -path \*python_env/bin/jupyter-nbconvert)
echo $nb_convert_path
git config filter.strip-notebook-output.clean "$nb_convert_path --ClearOutputPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR"
