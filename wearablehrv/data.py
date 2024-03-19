import pooch
import os
import shutil


# Define the URL to your GitHub repository's raw file access location
BASE_URL = "https://raw.githubusercontent.com/Aminsinichi/wearable-hrv/master/data/"

# Define the known file hashes for your files

FILE_HASHES = {
    "test_empatica.csv": "sha256:6c867057909380a5750fa9ec402aaf143556a094ac0ddd66d12021249e098fb0",
    "test_events.csv": "sha256:db0ffabd732bcf2ccfea10015e4e4ca5f78cadebc8dd833ee5a7005527a6c177",
    "test_heartmath.csv": "sha256:65beada0d914b927f61d9f7a4d744f2216fb01b0ac462afec7861c467d01e4c5",
    "test_kyto.csv": "sha256:a91e13a9a401989d0712f08ff4bb033c4d74ca0805c82b04f9d17daf432c7100",
    "test_rhythm.csv": "sha256:0cd5cca47a9ada9d07e205b754b7c0cdba8a704237b3f1dc70f95ff652bd8a64",
    "test_vu.txt": "sha256:45035a642a40c0489c7ee8d6e6297e202beef2b62f194b459bbd3c7ef37ab102",
    "P01.csv": "sha256:ba6606821be19724fe4555e2f5f124f30a7db8722cfeb79fd3f2389ab78e38f7",
    "P02.csv": "sha256:a8c78bc10979002a0739c0ff74c42d87989d813adbdcf97130e7d85789183e18",
    "P03.csv": "sha256:ff9e0fa85f67ff88d76ed9e4ca855d9675bdeaf5853524bdc299835e8fbf33c2",
    "P04.csv": "sha256:d54c87bdf277218693037ea6e69028fa7656fa36f912b85c7def854e9dda14dd",
    "P05.csv": "sha256:545dc0c3d770db8f19c04c3947292270a281a4b63648e6500bdebfa8b949bfe2",
    "P06.csv": "sha256:f0ea4d3519664eabd541973f32202a7dd6fe49d077eff92f5a4f9c25bbf2dcfe",
    "P07.csv": "sha256:b6699300fb15739e9e12914aec3ee4b43a3e550f2084294861dd852c4fb9818a",
    "P08.csv": "sha256:6fcb1e995563074e9fe3e37dff10bd3de4f6f8d6d9bf08e12260cd94fb513125",
    "P09.csv": "sha256:70bb209e2e9666eccd5be6d534841d7a6828a2ead7af88fce986d090cbe9469b",
    "P10.csv": "sha256:3cffb0ceae0c9c4eb45fb703655392aeb794e5e47f266c04525977f203b179ba",
}

# Create a Pooch object
data_fetcher = pooch.create(
    path=pooch.os_cache("wearablehrv"), base_url=BASE_URL, registry=FILE_HASHES
)


def download_data_and_get_path(file_names=None):
    """
    Fetch specified data files from the remote repository and store them locally.
    If no specific files are provided, all files defined in FILE_HASHES are downloaded.

    Parameters:
    - file_names (list, optional): A list of filenames to fetch. Defaults to None,
      which will fetch all files listed in FILE_HASHES.

    Returns:
    - str: The local directory path where the data files are stored.
    """
    # If no specific file names are provided, download all files listed in FILE_HASHES
    if file_names is None:
        file_names = FILE_HASHES.keys()

    for file_name in file_names:
        if file_name in FILE_HASHES:
            data_fetcher.fetch(file_name)
        else:
            print(
                f"Warning: {file_name} not found in FILE_HASHES and was not downloaded."
            )

    return pooch.os_cache("wearablehrv")


def clear_wearablehrv_cache():
    """
    Remove all files in the Pooch cache directory for the 'wearablehrv' package.
    The cache directory itself is not removed.
    """
    cache_dir = pooch.os_cache("wearablehrv")

    # Check if the directory exists
    if os.path.exists(cache_dir):
        # Iterate over all files and directories within the cache directory
        for filename in os.listdir(cache_dir):
            file_path = os.path.join(cache_dir, filename)
            try:
                # If it's a file or a symbolic link, delete it directly
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                # If it's a directory, delete the entire directory tree
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

        print("Cache cleared successfully.")
    else:
        print("Cache directory does not exist.")
