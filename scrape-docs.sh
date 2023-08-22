#!/bin/bash
wget -e robots=off --no-clobber --page-requisites \
  --html-extension --convert-links --restrict-file-names=windows \
  --domains docs.ray.io --no-parent --accept=html \
  -P $SCRAPE_PATH \
  https://docs.ray.io/en/master/
