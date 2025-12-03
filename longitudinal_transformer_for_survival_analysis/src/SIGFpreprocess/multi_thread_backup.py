#!/usr/bin/env python3
"""
Multi-threaded backup script for SIGF dataset.
Continues from where previous backup left off.
"""

import os
import shutil
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import List, Tuple


def get_missing_files(source_dir: str, dest_dir: str) -> List[Tuple[str, str]]:
    """Find files that exist in source but not in destination."""
    missing_files = []
    
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            source_path = os.path.join(root, file)
            # Calculate relative path from source_dir
            rel_path = os.path.relpath(source_path, source_dir)
            dest_path = os.path.join(dest_dir, rel_path)
            
            # Check if file exists in destination
            if not os.path.exists(dest_path):
                missing_files.append((source_path, dest_path))
            elif os.path.getsize(source_path) != os.path.getsize(dest_path):
                # File exists but size differs, need to recopy
                missing_files.append((source_path, dest_path))
    
    return missing_files


def copy_file_safe(source_path: str, dest_path: str) -> bool:
    """Safely copy a single file with directory creation."""
    try:
        # Create destination directory if it doesn't exist
        dest_dir = os.path.dirname(dest_path)
        os.makedirs(dest_dir, exist_ok=True)
        
        # Copy file
        shutil.copy2(source_path, dest_path)
        
        # Verify copy
        if os.path.exists(dest_path) and os.path.getsize(source_path) == os.path.getsize(dest_path):
            return True
        else:
            print(f"Copy verification failed: {source_path}")
            return False
            
    except Exception as e:
        print(f"Error copying {source_path}: {str(e)}")
        return False


def copy_files_threaded(file_pairs: List[Tuple[str, str]], max_workers: int = 8) -> None:
    """Copy files using multiple threads."""
    total_files = len(file_pairs)
    completed = 0
    failed = []
    
    print(f"Starting multi-threaded copy of {total_files} files with {max_workers} workers...")
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all copy tasks
        future_to_files = {
            executor.submit(copy_file_safe, src, dst): (src, dst) 
            for src, dst in file_pairs
        }
        
        # Process completed tasks
        for future in as_completed(future_to_files):
            src, dst = future_to_files[future]
            completed += 1
            
            try:
                success = future.result()
                if success:
                    if completed % 100 == 0 or completed == total_files:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed
                        print(f"Progress: {completed}/{total_files} ({completed/total_files*100:.1f}%) "
                              f"Rate: {rate:.1f} files/sec")
                else:
                    failed.append((src, dst))
                    
            except Exception as e:
                print(f"Task failed for {src}: {str(e)}")
                failed.append((src, dst))
    
    elapsed_time = time.time() - start_time
    print(f"\nCompleted in {elapsed_time:.1f} seconds")
    print(f"Successfully copied: {completed - len(failed)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print("\nFailed files:")
        for src, dst in failed[:10]:  # Show first 10 failures
            print(f"  {src}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")


def main():
    source_dir = "/home/lin01231/public/datasets/SIGF"
    dest_dir = "/home/lin01231/public/datasets/SIGF_backup"
    
    print("SIGF Dataset Multi-threaded Backup Tool")
    print("=" * 50)
    
    # Check if source exists
    if not os.path.exists(source_dir):
        print(f"Error: Source directory not found: {source_dir}")
        return
    
    # Create destination if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    print("Scanning for missing files...")
    missing_files = get_missing_files(source_dir, dest_dir)
    
    if not missing_files:
        print("Backup is already complete! No files to copy.")
        return
    
    print(f"Found {len(missing_files)} files to copy")
    
    # Calculate total size of missing files
    total_size = 0
    for src, _ in missing_files:
        try:
            total_size += os.path.getsize(src)
        except:
            pass
    
    print(f"Total size to copy: {total_size / (1024*1024):.1f} MB")
    
    # Ask for confirmation
    response = input("Proceed with multi-threaded copy? (y/N): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Ask for thread count
    try:
        max_workers = int(input("Enter number of threads (default 8): ") or "8")
        max_workers = max(1, min(max_workers, 16))  # Limit between 1-16
    except ValueError:
        max_workers = 8
    
    print(f"Using {max_workers} threads")
    
    # Start copying
    copy_files_threaded(missing_files, max_workers)
    
    print("\nVerifying final backup...")
    final_missing = get_missing_files(source_dir, dest_dir)
    
    if not final_missing:
        print("✅ Backup completed successfully!")
        
        # Show final sizes
        src_size = sum(os.path.getsize(os.path.join(root, file)) 
                      for root, _, files in os.walk(source_dir) 
                      for file in files) / (1024*1024)
        dst_size = sum(os.path.getsize(os.path.join(root, file)) 
                      for root, _, files in os.walk(dest_dir) 
                      for file in files) / (1024*1024)
        
        print(f"Source size: {src_size:.1f} MB")
        print(f"Backup size: {dst_size:.1f} MB")
    else:
        print(f"⚠️  {len(final_missing)} files still missing after backup")


if __name__ == "__main__":
    main()