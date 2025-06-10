#!/usr/bin/env sh

version=$(cat /app/model_storage/current)
cp /app/model_storage/$version/model.mpk /app/model.mpk

