#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <download_directory>"
    exit 1
fi

download_dir="$1"
download_dir=$(realpath "$download_dir")

if [ ! -d "$download_dir" ]; then
    mkdir -p "$download_dir"
fi

# tail -n +3 is used to skip the first two lines of the file.
tail -n +3 download_data.csv | while IFS= read -r line; do
    # Extract the dataset name from the URL
    dataset_name=$(echo "$line" | rev | cut -d'/' -f1 | rev | cut -d'.' -f1)

    # Use wget to download the dataset
    echo "Downloading $dataset_name..."
    wget -P "$download_dir" -O "${download_dir}/${dataset_name}.tar" "$line"

    # Extract the dataset
    echo "Extracting $dataset_name..."
    tar -xvf "${download_dir}/${dataset_name}.tar" -C "$download_dir"

    # Remove the downloaded .tar file
    echo "Cleaning up $dataset_name.tar..."
    rm -rf "${download_dir}/${dataset_name}.tar"

    echo "$dataset_name processing complete."
done

script_dir=$(dirname "$(realpath "$0")")
data_dir="${script_dir}/data"

if [ "$(realpath "$data_dir")" != "$download_dir" ]; then
    if [ -e "$data_dir" ]; then
        echo "Error: directory $data_dir already exists. Not creating a symlink."
    else
        ln -s "$download_dir" "$data_dir"
        echo "Symlink created from $data_dir to $download_dir."
    fi
fi