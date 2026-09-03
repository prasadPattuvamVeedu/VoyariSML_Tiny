from pathlib import Path
import kagglehub


PROJECT_ROOT = Path(r"D:\voyari_sml")

RAW_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "VoyariLM_DATA"
    / "01_RAW_SOURCES"
)

DATASET_ID = (
    "dhrubangtalukdar/"
    "top-indian-places-to-visit-indian-tourism"
)

OUTPUT_FOLDER = (
    RAW_DATA_DIR
    / "top_indian_places"
)

RAW_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print()
print("=" * 70)
print("DOWNLOADING TOP INDIAN PLACES DATASET")
print("=" * 70)

print("Dataset:")
print(DATASET_ID)

print()
print("Destination:")
print(OUTPUT_FOLDER)


downloaded_path = kagglehub.dataset_download(
    DATASET_ID,
    output_dir=str(OUTPUT_FOLDER),
)

downloaded_path = Path(downloaded_path)


print()
print("=" * 70)
print("DOWNLOAD COMPLETE")
print("=" * 70)

print("Saved to:")
print(downloaded_path)

print()
print("Files:")

for file_path in downloaded_path.rglob("*"):

    if file_path.is_file():

        size_mb = (
            file_path.stat().st_size
            / 1024
            / 1024
        )

        print(
            f"{file_path.name:45} "
            f"{size_mb:.2f} MB"
        )

print("=" * 70)