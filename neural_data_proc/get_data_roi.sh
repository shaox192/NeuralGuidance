#!/bin/bash

####################
# We need these files:
# Kastner 2015 for ROI
# nsdgeneral for Correction
# subject T1 volume for visualization
###
# online_sub: sub ID used in NSD
# local_sub: sub ID we want to use
####################

online_sub=$1

if [ -z "$2" ]; then
    local_sub=$online_sub  # If $2 is not provided, set local_sub to online_sub
else
    local_sub=$2
fi

echo "online_sub: $online_sub"
echo "local_sub: $local_sub"

dest_dir=${local_sub}_nsd
if [ ! -d $dest_dir ] 
then
    echo making $dest_dir ...
    mkdir $dest_dir
fi

src=https://natural-scenes-dataset.s3.amazonaws.com/nsddata/ppdata/${online_sub}/func1pt8mm

wget ${src}/roi/Kastner2015.nii.gz -P $dest_dir
wget ${src}/roi/nsdgeneral.nii.gz -P $dest_dir
wget ${src}/T1_to_func1pt8mm.nii.gz -P $dest_dir
wget https://natural-scenes-dataset.s3.amazonaws.com/nsddata/experiments/nsd/nsd_expdesign.mat -P $dest_dir