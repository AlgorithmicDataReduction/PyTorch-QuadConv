#!/bin/bash

# specify the directory containing the files
if [ $# -eq 0 ]; then
    echo "Error: No directory provided."
    exit 1
fi

directory="$1"

echo $directory

# loop through all files in the directory
for file in $directory/*.sh
do 

    # check if the file is a shell script and execute it
    if [ -x "$file" ]; then
        echo "Running $file"
        bsub $file
    fi

done