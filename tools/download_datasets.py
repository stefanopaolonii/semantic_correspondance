import os
import argparse
import tempfile
import shutil
import pooch


def download_spair(datasets_dir):
    spair_folder = os.path.join(datasets_dir, "SPair-71k")
    
    if os.path.exists(spair_folder):
        print("SPair-71k already exists, skipping download")
        return
    
    spair_url = "http://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz"
    pooch.retrieve(
        url=spair_url,
        known_hash=None,
        fname="SPair-71k.tar.gz",
        path=datasets_dir,
        processor=pooch.Untar(extract_dir="."),
        progressbar=True
    )
    
    print("SPair-71k dataset ready")


def download_pfpascal(datasets_dir):
    pfpascal_folder = os.path.join(datasets_dir, "pf-pascal")
    
    if os.path.exists(pfpascal_folder):
        print("PF-PASCAL already exists, skipping download")
        return
    
    os.makedirs(pfpascal_folder, exist_ok=True)
    
    pfpascal_dataset_url = "https://www.di.ens.fr/willow/research/proposalflow/dataset/PF-dataset-PASCAL.zip"
    pooch.retrieve(
        url=pfpascal_dataset_url,
        known_hash=None,
        fname="PF-dataset-PASCAL.zip",
        path=pfpascal_folder,
        processor=pooch.Unzip(extract_dir="."),
        progressbar=True
    )
    
    pairs_url = "https://www.robots.ox.ac.uk/~xinghui/sd4match/pf-pascal_image_pairs.zip"
    temp_dir = tempfile.mkdtemp()
    pooch.retrieve(
        url=pairs_url,
        known_hash=None,
        fname="pf-pascal_image_pairs.zip",
        path=temp_dir,
        processor=pooch.Unzip(extract_dir="."),
        progressbar=True
    )
    
    csv_source = os.path.join(temp_dir, "pf-pascal_image_pairs")
    if os.path.exists(csv_source):
        for csv_file in ["trn_pairs.csv", "val_pairs.csv", "test_pairs.csv"]:
            src = os.path.join(csv_source, csv_file)
            dst = os.path.join(pfpascal_folder, csv_file)
            if os.path.exists(src):
                shutil.move(src, dst)
    
    shutil.rmtree(temp_dir, ignore_errors=True)
    zip_path = os.path.join(pfpascal_folder, "PF-dataset-PASCAL.zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)
    
    print("PF-PASCAL dataset ready")


def download_pfwillow(datasets_dir):
    pfwillow_folder = os.path.join(datasets_dir, "pf-willow")
    
    if os.path.exists(pfwillow_folder):
        print("PF-WILLOW already exists, skipping download")
        return
    
    os.makedirs(pfwillow_folder, exist_ok=True)
    
    pfwillow_url = "https://www.di.ens.fr/willow/research/proposalflow/dataset/PF-dataset.zip"
    pooch.retrieve(
        url=pfwillow_url,
        known_hash=None,
        fname="PF-dataset.zip",
        path=pfwillow_folder,
        processor=pooch.Unzip(extract_dir="."),
        progressbar=True
    )
    
    pairs_url = "https://www.robots.ox.ac.uk/~xinghui/sd4match/test_pairs.csv"
    pooch.retrieve(
        url=pairs_url,
        known_hash=None,
        fname="test_pairs.csv",
        path=pfwillow_folder,
        progressbar=True
    )
    
    zip_path = os.path.join(pfwillow_folder, "PF-dataset.zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)
    
    print("PF-WILLOW dataset ready")


def main():
    parser = argparse.ArgumentParser(
        description="Download benchmark datasets for semantic correspondence"
    )
    parser.add_argument(
        "--dataset", 
        choices=["spair", "pfpascal", "pfwillow", "all"], 
        default="all", 
        help="Dataset to download (default: all)"
    )
    args = parser.parse_args()
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    datasets_dir = os.path.join(project_root, "asset", "dataset")
    os.makedirs(datasets_dir, exist_ok=True)
    
    if args.dataset == "all":
        download_spair(datasets_dir)
        download_pfpascal(datasets_dir)
        download_pfwillow(datasets_dir)
    elif args.dataset == "spair":
        download_spair(datasets_dir)
    elif args.dataset == "pfpascal":
        download_pfpascal(datasets_dir)
    elif args.dataset == "pfwillow":
        download_pfwillow(datasets_dir)
    
if __name__ == "__main__":
    main()
