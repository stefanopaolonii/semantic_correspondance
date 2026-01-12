import os
import argparse
import pooch


def download_file(url, folder, fname=None):
    if fname is None:
        url_path = url.split('?')[0]  
        fname = os.path.basename(url_path)
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_dir = os.path.join(project_root, "asset", "weights")
    dest_folder = os.path.join(weights_dir, folder)
    dest_path = os.path.join(dest_folder, fname)
    
    if os.path.exists(dest_path):
        print(f"File '{fname}' already exists in '{folder}'. Skipping.")
        return
    
    print(f"Downloading {fname}")
    pooch.retrieve(
        url=url,
        known_hash=None,
        fname=fname,
        path=dest_folder,
        progressbar=True
    )


def download_dinov2():
    weights = {
        "small": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
        "base": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
        "large": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth"
    }
    
    for variant, url in weights.items():
        download_file(url, "dinov2")


def download_dinov3():
    print("DINOv3 requires manual download (Meta AI authorization)")
    print("→ https://github.com/facebookresearch/dinov3")
    print("→ Place weights in: asset/weights/dinov3/")


def download_sam():
    weights = {
        "base": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "large": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "huge": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    }
    
    for variant, url in weights.items():
        download_file(url, "sam")


def download_clip():
    weights = {
        "ViT-B-32.pt": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
        "ViT-B-16.pt": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
        "ViT-L-14.pt": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    }
    
    for fname, url in weights.items():
        download_file(url, "clip", fname=fname)


def main():
    parser = argparse.ArgumentParser(
        description="Download pretrained model weights"
    )
    parser.add_argument(
        "--model", 
        choices=["dinov2", "dinov3", "sam", "clip", "all"], 
        default="all", 
        help="Model weights to download (default: all)"
    )
    args = parser.parse_args()
    
    if args.model == "all":
        download_dinov2()
        download_dinov3()
        download_sam()
        download_clip()
    elif args.model == "dinov2":
        download_dinov2()
    elif args.model == "dinov3":
        download_dinov3()
    elif args.model == "sam":
        download_sam()
    elif args.model == "clip":
        download_clip()
    
if __name__ == "__main__":
    main()
