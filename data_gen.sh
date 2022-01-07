#!/bin/bash
N=30
mkdir -p test
cd test
mkdir -p with_mask
mkdir -p without_mask
cd ..
mv test/with_mask/* data/with_mask/
mv test/without_mask/* data/without_mask/
cd data/with_mask
ls |sort -R |tail -$N |while read file; do
    mv $file ../../test/with_mask
done
cd ../..
cd data/without_mask
ls |sort -R |tail -$N |while read file; do
    mv $file ../../test/without_mask
done
cd ..
