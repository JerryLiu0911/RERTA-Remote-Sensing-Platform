import os

def compare_folders(folder1, folder2):
    # Get list of files in each folder and remove extensions
    files1 = set()
    files2 = set()
    
    for f in os.listdir(folder1):
        item_path = os.path.join(folder1, f)
        if os.path.isdir(item_path):
            name_without_ext = os.path.splitext(f)[0]
            files1.add(name_without_ext)

    for f in os.listdir(folder2):
        name_without_ext = os.path.splitext(f)[0]  # Remove extension
        files2.add(name_without_ext)

    # Files only in folder1
    only_in_folder1 = files1 - files2
    # Files only in folder2
    only_in_folder2 = files2 - files1

    # Print results
    if only_in_folder1:
        print(f"Files only in {folder1}:")
        for f in sorted(only_in_folder1):
            print(f"  {f}")
    else:
        print(f"No files unique to {folder1}")

    if only_in_folder2:
        print(f"\nFiles only in {folder2}:")
        for f in sorted(only_in_folder2):
            print(f"  {f}")
    else:
        print(f"No files unique to {folder2}")

    # Summary
    print(f"\nTotal files in {folder1}: {len(files1)}")
    print(f"Total files in {folder2}: {len(files2)}")
    print(f"Discrepancies found: {len(only_in_folder1) + len(only_in_folder2)}")



# folder1 = r"D:\Jerry\Lidar Data from Becky\All-scans"
# folder2 = r"D:\Jerry\Lidar Data from Becky\Processed Data Rerta smallholders 2nd try"

# compare_folders(folder1, folder2)

def add_file_extension_fls(folder_path):
    files_renamed = 0
    errors = 0
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Only process files, not directories
        if os.path.isdir(file_path):
            # Check if file already has .fls extension
            if not filename.endswith('.fls'):
                new_filename = filename + '.fls'
                new_file_path = os.path.join(folder_path, new_filename)
                
                try:
                    os.rename(file_path, new_file_path)
                    print(f"✅ Renamed: {filename} → {new_filename}")
                    files_renamed += 1
                except Exception as e:
                    print(f"❌ Error renaming {filename}: {e}")
                    errors += 1
            else:
                print(f"⏭️  Skipped: {filename} (already has .fls extension)")
    
    print("-" * 60)
    print(f"Summary:")
    print(f"  Files renamed: {files_renamed}")
    print(f"  Errors: {errors}")
    print(f"  Total files processed: {files_renamed + errors}")
add_file_extension_fls(r"D:\Jerry\Lidar Data from Becky\LiDAR\LiDAR-Scans-2025-02")