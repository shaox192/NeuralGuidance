#!/bin/bash

####################
# We need at least:
# Kastner 2015
# nsdgeneral
####################

online_sub=$1
local_sub=$2

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