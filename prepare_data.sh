#!/bin/bash

mkdir "data" && wget "https://russiansuperglue.com/tasks/download" --no-check-certificate -O "combined.zip" && unzip "combined.zip" "combined/*" -d "data/" && rm "combined.zip"
