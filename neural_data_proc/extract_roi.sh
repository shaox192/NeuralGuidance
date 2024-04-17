#!/bin/bash

####################
# Kastner
# 1 V1v	    2 V1d	   
# 3 V2v	    4 V2d	   
# 5 V3v	    6 V3d	    
# 7 hV4	  
# 8 VO1	    9 VO2	    
# 10 PHC1	11 PHC2	    
# 12 TO2	13 TO1	   
# 14 LO2	15 LO1	   
# 16 V3B	17 V3A	 
####################

sub=$1
roi=$2
sub_dir=${sub}_orig_roi
out_dir=roi

echo "====> Extracting $roi from subject $sub"

if [[ "$roi" == "V1" ]]
then
	echo "----> entering V1"
	3dcalc -a ${sub_dir}/Kastner2015.nii.gz \
	       -expr "equals(a, 1) + equals(a, 2)" \
	       -prefix ${sub_dir}/V1

elif [[ "$roi" == "V2" ]]
then
	echo "entering V2"
	3dcalc -a ${sub_dir}/Kastner2015.nii.gz \
	       -expr "equals(a, 3) + equals(a, 4)" \
	       -prefix ${sub_dir}/V2

elif [[ "$roi" == "V1v" ]]
then
	echo "entering V1v"
	3dcalc -a ${sub_dir}/Kastner2015.nii.gz \
	       -expr "equals(a, 1)" \
	       -prefix ${sub_dir}/V1v

elif [[ "$roi" == "V1d" ]]
then
	echo "entering V1d"
	3dcalc -a ${sub_dir}/Kastner2015.nii.gz \
	       -expr "equals(a, 2)" \
	       -prefix ${sub_dir}/V1d

elif [[ "$roi" == "V2v" ]]
then
	echo "entering V2v"
	3dcalc -a ${sub_dir}/Kastner2015.nii.gz \
	       -expr "equals(a, 3)" \
	       -prefix ${sub_dir}/V2v

elif [[ "$roi" == "V2d" ]]
then
	echo "entering V2d"
	3dcalc -a ${sub_dir}/Kastner2015.nii.gz \
	       -expr "equals(a, 4)" \
	       -prefix ${sub_dir}/V2d

elif [[ "$roi" == "V4" ]]
then
	echo "entering V4"
	3dcalc -a ${sub_dir}/Kastner2015.nii.gz \
	       -expr "equals(a, 7)" \
	       -prefix ${sub_dir}/V4

elif [[ "$roi" == "LO" ]]
then
	echo "entering LO"
	3dcalc -a ${sub_dir}/Kastner2015.nii.gz \
	       -expr "equals(a, 14) + equals(a, 15)" \
	       -prefix ${sub_dir}/LO

elif [[ "$roi" == "VO" ]]
then
	echo "entering VO"
	3dcalc -a ${sub_dir}/Kastner2015.nii.gz \
	       -expr "equals(a, 8) + equals(a, 9)" \
	       -prefix ${sub_dir}/VO

elif [[ "$roi" == "TO" ]]
then
	echo "entering TO"
	3dcalc -a ${sub_dir}/Kastner2015.nii.gz \
	       -expr "equals(a, 12) + equals(a, 13)" \
	       -prefix ${sub_dir}/TO

elif [[ "$roi" == "PHC" ]]
then
	echo "entering PHC"
	3dcalc -a ${sub_dir}/Kastner2015.nii.gz \
	       -expr "equals(a, 10) + equals(a, 11)" \
	       -prefix ${sub_dir}/PHC

else
	echo "Enter a valid ROI"
fi



3dcalc -a ${sub_dir}/${roi}+orig \
	   -b ${sub_dir}/nsdgeneral.nii.gz \
	   -expr "a * b" \
	   -prefix ${sub_dir}/${sub}_${roi}_roi


3dAFNItoNIFTI -prefix ${sub_dir}/${sub}_${roi}_roi \
			  ${sub_dir}/${sub}_${roi}_roi+orig

mv ${sub_dir}/*.nii $out_dir



