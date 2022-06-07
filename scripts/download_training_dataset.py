import os
from shutil import copyfile, copytree
import glob
import shutil

shutil.rmtree("../tmp_download_files")

os.makedirs("../tmp_download_files", exist_ok=True)

# MSRA10K
os.system("wget -P ../tmp_download_files http://mftp.mmcheng.net/Data/MSRA10K_Imgs_GT.zip")
# ECSSD_url 答案要重新下載
os.system(
    "wget -P ../tmp_download_files http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/data/ECSSD/ground_truth_mask.zip")
os.system("wget -P ../tmp_download_files http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/data/ECSSD/images.zip")
# FSS1000  這行需要另外下載
os.system(
    "wget --load-cookies /tmp/cookies.txt \"https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=16TgqOeI_0P41Eh3jWQlxlRXG9KIqtMgI' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p')&id=16TgqOeI_0P41Eh3jWQlxlRXG9KIqtMgI\" -O ../tmp_download_files/fewshot_data.zip && rm -rf /tmp/cookies.txt")
# DUT-OMRON ========== Link is not working???
os.system("wget -P ../tmp_download_files http://saliencydetection.net/duts/download/DUTS-TR.zip")
os.system("wget -P ../tmp_download_files http://saliencydetection.net/duts/download/DUTS-TE.zip")

# Unzip everything
os.system("PowerShell Expand-Archive -Path \"../tmp_download_files/MSRA10K_Imgs_GT.zip\" -DestinationPath \"../tmp_download_files\"")
os.system("PowerShell Expand-Archive -Path \"../tmp_download_files/images.zip\" -DestinationPath \"../tmp_download_files\"")
os.system("PowerShell Expand-Archive -Path \"../tmp_download_files/ground_truth_mask.zip\" -DestinationPath \"../tmp_download_files\"")
os.system("PowerShell Expand-Archive -Path \"../tmp_download_files/fewshot_data.zip\" -DestinationPath \"../tmp_download_files\"")

os.makedirs("../tmp_download_files/DUTS", exist_ok=True)
os.system("PowerShell Expand-Archive -Path \"../tmp_download_files/DUTS-TR.zip\" -DestinationPath \"../tmp_download_files/DUTS\"")
os.system("PowerShell Expand-Archive -Path \"../tmp_download_files/DUTS-TE.zip\" -DestinationPath \"../tmp_download_files/DUTS\"")

# Move to data folder
os.makedirs("../data/DUTS-TE", exist_ok=True)
os.makedirs("../data/DUTS-TR", exist_ok=True)

for file in glob.glob("../tmp_download_files/DUTS/DUTS-TE/*/*"):
    copyfile(file, "../data/DUTS-TE/" + os.path.basename(file))

for file in glob.glob("../tmp_download_files/DUTS/DUTS-TR/*/*"):
    copyfile(file, "../data/DUTS-TR/" + os.path.basename(file))

os.makedirs("../data/fss", exist_ok=True)
for cl in os.listdir("../tmp_download_files/fewshot_data/fewshot_data"):
    copytree("../tmp_download_files/fewshot_data/fewshot_data/" + cl, "../data/fss/" + cl)

os.makedirs("../data/ecssd", exist_ok=True)
for gt in glob.glob("../tmp_download_files/images/*"):
    copyfile(gt, "../data/ecssd/{}".format(os.path.basename(gt)))
for gt in glob.glob("../training_dataset/ground_truth_mask/*"):
    copyfile(gt, "../data/ecssd/{}".format(os.path.basename(gt)))

os.makedirs("../data/MSRA_10K", exist_ok=True)
for gt in glob.glob("../tmp_download_files/MSRA10K_Imgs_GT/Imgs/*"):
    copyfile(gt, "../data/MSRA_10K/{}".format(os.path.basename(gt)))

# Deleted temp files
shutil.rmtree("../tmp_download_files")