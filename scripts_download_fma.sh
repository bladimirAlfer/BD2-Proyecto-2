#!/bin/bash

set -e

mkdir -p data/raw/fma
cd data/raw/fma

echo "Descargando metadata FMA..."
curl -L -O https://os.unil.cloud.switch.ch/fma/fma_metadata.zip

echo "Descargando FMA Small..."
curl -L -O https://os.unil.cloud.switch.ch/fma/fma_small.zip

echo "Verificando SHA1..."
echo "f0df49ffe5f2a6008d7dc83c6915b31835dfe733  fma_metadata.zip" | shasum -a 1 -c -
echo "ade154f733639d52e35e32f5593efe5be76c6d70  fma_small.zip" | shasum -a 1 -c -

echo "Descomprimiendo..."
unzip -q fma_metadata.zip
unzip -q fma_small.zip

echo "FMA listo."
