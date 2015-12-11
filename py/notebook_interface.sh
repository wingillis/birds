#!/bin/bash
if ! [ -s ${PWD##*/}.ipynb ];
then
  cp ~/Documents/MATLAB/birds/py/notebook_template.ipynb ${PWD##*/}.ipynb
else
  echo "Notebook exists"
fi
jupyter notebook ${PWD##*/}.ipynb
