#!/usr/bin/env python3
"""
Clean up ImageNet download directory before re-downloading
"""
import shutil
from pathlib import Path
from tqdm import tqdm

def cleanup_directory(base_dir: Path, keep_zips: bool = False):
    """Remove all files and directories except zip files if keep_zips=True"""
    if not base_dir.exists():
        print(f"Directory {base_dir} does not exist. Nothing to clean.")
        return
    
    print(f"Cleaning up {base_dir}...")
    
    items = list(base_dir.iterdir())
    if not items:
        print("Directory is already empty.")
        return
    
    # Count items
    dirs = [item for item in items if item.is_dir()]
    files = [item for item in items if item.is_file()]
    zip_files = [f for f in files if f.suffix == '.zip']
    
    print(f"Found:")
    print(f"  - {len(dirs)} directories")
    print(f"  - {len(files)} files ({len(zip_files)} zip files)")
    
    if keep_zips and zip_files:
        print(f"\nKeeping {len(zip_files)} zip file(s) for re-extraction...")
    
    # Remove directories
    if dirs:
        print(f"\nRemoving {len(dirs)} directories...")
        for d in tqdm(dirs, desc="Removing dirs"):
            try:
                shutil.rmtree(d)
            except Exception as e:
                print(f"Warning: Could not remove {d}: {e}")
    
    # Remove files (except zips if keep_zips=True)
    files_to_remove = [f for f in files if not (keep_zips and f.suffix == '.zip')]
    if files_to_remove:
        print(f"\nRemoving {len(files_to_remove)} files...")
        for f in tqdm(files_to_remove, desc="Removing files"):
            try:
                f.unlink()
            except Exception as e:
                print(f"Warning: Could not remove {f}: {e}")
    
    print("\nCleanup completed!")

def main():
    base_dir = Path("/mnt/imagenet")
    
    print("=" * 70)
    print("ImageNet Directory Cleanup")
    print("=" * 70)
    
    response = input(f"\nThis will DELETE ALL DATA in {base_dir}\n"
                     f"Are you sure you want to continue? (yes/no): ")
    
    if response.lower() != 'yes':
        print("Cleanup cancelled.")
        return
    
    keep_zips = input("\nKeep zip files for re-extraction? (yes/no) [default: no]: ")
    keep_zips = keep_zips.lower() == 'yes'
    
    cleanup_directory(base_dir, keep_zips=keep_zips)
    
    print("\n" + "=" * 70)
    print("Ready for fresh download!")
    print("=" * 70)

if __name__ == "__main__":
    main()

