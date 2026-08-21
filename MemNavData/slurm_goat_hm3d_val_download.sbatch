#!/usr/bin/env bash
# Download the licensed HM3D v0.2 validation Habitat archive without exposing
# the Matterport API token in argv, the Slurm environment, or logs.
#SBATCH --job-name=goat_hm3d_val
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
umask 0022

readonly ROOT="${GOAT_HM3D_ROOT:-/scratch/yz11502/Research/datasets/goat_bench_20260814}"
readonly CREDENTIALS="${MATTERPORT_NETRC:-${HOME}/.config/hm3d/netrc}"
readonly URL="https://api.matterport.com/resources/habitat/hm3d-val-habitat-v0.2.tar"
readonly DOWNLOAD_DIR="${ROOT}/downloads"
readonly RECEIPT_DIR="${ROOT}/receipts"
readonly ARCHIVE="${DOWNLOAD_DIR}/hm3d-val-habitat-v0.2.tar"
readonly PARTIAL="${ARCHIVE}.part"
readonly MEMBER_LIST="${RECEIPT_DIR}/hm3d-val-habitat-v0.2.members.txt"
readonly CHECKSUM="${RECEIPT_DIR}/hm3d-val-habitat-v0.2.sha256"
readonly RECEIPT="${RECEIPT_DIR}/hm3d-val-habitat-v0.2.download.json"

mkdir -p "${DOWNLOAD_DIR}" "${RECEIPT_DIR}"
test -r "${CREDENTIALS}"

python3 - "${CREDENTIALS}" <<'PY'
import netrc
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if path.stat().st_mode & 0o077:
    raise SystemExit("Matterport credential file must not be group/world accessible")
auth = netrc.netrc(str(path)).authenticators("api.matterport.com")
if auth is None or not auth[0] or not auth[2]:
    raise SystemExit("Matterport token ID/secret are absent")
print("credential_preflight=ok")
PY

exec 9>"${ROOT}/hm3d-val-habitat-v0.2.download.lock"
flock -n 9 || {
  echo "another HM3D validation download owns the lock" >&2
  exit 3
}

if [[ -f "${ARCHIVE}" ]]; then
  echo "archive_already_present=yes"
else
  curl --fail --location --silent --show-error \
    --retry 8 --retry-delay 5 --retry-all-errors \
    --connect-timeout 30 --continue-at - \
    --netrc-file "${CREDENTIALS}" \
    --output "${PARTIAL}" "${URL}"
  file -b "${PARTIAL}" | grep -q 'tar archive'
  mv "${PARTIAL}" "${ARCHIVE}"
fi

member_tmp="${MEMBER_LIST}.tmp.${SLURM_JOB_ID:-local}"
tar -tf "${ARCHIVE}" > "${member_tmp}"
mv "${member_tmp}" "${MEMBER_LIST}"

checksum_tmp="${CHECKSUM}.tmp.${SLURM_JOB_ID:-local}"
sha256sum "${ARCHIVE}" > "${checksum_tmp}"
mv "${checksum_tmp}" "${CHECKSUM}"

python3 - "${ARCHIVE}" "${MEMBER_LIST}" "${CHECKSUM}" "${RECEIPT}" <<'PY'
import datetime
import json
import pathlib
import sys

archive, members, checksum, receipt = map(pathlib.Path, sys.argv[1:])
names = members.read_text().splitlines()
sha256 = checksum.read_text().split()[0]
payload = {
    "schema_version": "goat_hm3d_val_download_v1_20260814",
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "archive": str(archive.resolve()),
    "archive_bytes": archive.stat().st_size,
    "archive_sha256": sha256,
    "tar_member_count": len(names),
    "glb_members": sum(name.endswith(".glb") for name in names),
    "navmesh_members": sum(name.endswith(".navmesh") for name in names),
    "semantic_members": sum("semantic" in name.lower() for name in names),
    "auth_secret_recorded": False,
    "complete": True,
}
receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, sort_keys=True))
PY
