#!/bin/bash
pip install --upgrade pip setuptools wheel
pip install --only-binary=all --no-cache-dir -r requirements.txt
