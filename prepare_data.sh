#!/bin/bash

mkdir data

wget "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/AX-b.zip" && unzip AX-b.zip -d "data/" && rm "AX-b.zip"
wget "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/CB.zip" && unzip "CB.zip" -d "data/" && rm "CB.zip"
wget "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/COPA.zip" && unzip "COPA.zip" -d "data/" && rm "COPA.zip"
wget "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/MultiRC.zip" && unzip "MultiRC.zip" -d "data/" && rm "MultiRC.zip"
wget "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/RTE.zip" && unzip "RTE.zip" -d "data/" && rm "RTE.zip"
wget "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/WiC.zip" && unzip "WiC.zip" -d "data/" && rm "WiC.zip"
wget "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/WSC.zip" && unzip "WSC.zip" -d "data/" && rm "WSC.zip"
wget "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/BoolQ.zip" && unzip "BoolQ.zip" -d "data/" && rm "BoolQ.zip"
wget "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/ReCoRD.zip" && unzip "ReCoRD.zip" -d "data/" && rm "ReCoRD.zip"
wget "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/AX-g.zip" && unzip "AX-g.zip" -d "data/" && rm "AX-g.zip"
