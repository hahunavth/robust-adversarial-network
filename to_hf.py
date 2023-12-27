from huggingface_hub import HfApi
api = HfApi()

REPO_ID = "hahunavth/robust-adversarial-network"
if not api.repo_exists(REPO_ID):
    api.create_repo(REPO_ID)

# upload folder
api.upload_folder(
    repo_id=REPO_ID,
    folder_path='./snapshots',
    path_in_repo="snapshots",
)

api.upload_folder(
    repo_id=REPO_ID,
    folder_path='./logs',
    path_in_repo="tensorboard",
)

# api.upload_file(
#     path_or_fileobj="/path/to/local/folder/README.md",
#     path_in_repo="README.md",
#     repo_id="username/test-dataset",
#     repo_type="dataset",
# )