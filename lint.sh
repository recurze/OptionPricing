#!/bin/sh

pyfiles=`git ls-files '*.py'`
flake8 --max-line-length 120 $pyfiles
mypy --ignore-missing-imports $pyfiles
