#!/usr/bin/env bash

DIR=data/molecules/
#mkdir -p $DIR
cd $DIR


FILE=aqsol_graph_raw.zip
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl "https://www.dropbox.com/s/lzu9lmukwov12kt/aqsol_graph_raw.zip?dl=1" -o aqsol_graph_raw.zip -J -L -k
	unzip aqsol_graph_raw.zip -d ./
fi


