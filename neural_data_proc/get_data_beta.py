import subprocess
import argparse
import os
import utils

parser = argparse.ArgumentParser(description='download beta files')
parser.add_argument('--sub-onl', type=str, help='subj01, subj02, ...')
parser.add_argument('--sub-loc', type=str, help='sub1, sub2, ...')
parser.add_argument('--num-ses', type=int, default=40, help='how many beta session to download')
args = parser.parse_args()
utils.show_input_args(args)

sub = args.sub_onl
sub_dest = args.sub_loc

dest_dir = f"./{sub_dest}_betas"
print(f"Destination folder: {dest_dir}")
if not os.path.exists(dest_dir):
    print(f"dest dir: {dest_dir} not exist, making... ")
    os.mkdir(dest_dir)

def runcmd(cmd, verbose = False, *args, **kwargs):

    process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        shell = True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass


prefix = f"https://natural-scenes-dataset.s3.amazonaws.com/nsddata_betas/ppdata/{sub}/func1pt8mm/betas_fithrf_GLMdenoise_RR"
for i in range(1, args.num_ses + 1):
    ses_num = f"0{i}" if i < 10 else str(i)
    print(f"\nDownloading for session{ses_num} ...")
    runcmd(f'wget {prefix}/betas_session{ses_num}.nii.gz -P {dest_dir}/', verbose = False)
    print(f"DONE: session{ses_num}")
