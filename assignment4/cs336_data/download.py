import gzip
import shutil
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def get_directory_size_bytes(directory: Path) -> int:
    total_size = 0
    for path in directory.rglob("*"):
        if path.is_file():
            try:
                total_size += path.stat().st_size
            except OSError:
                pass
    return total_size


def download_one_file(file_url: str, target_path: Path, timeout: int):
    file_name = target_path.name
    temp_path = target_path.with_name(f"{file_name}.part")

    if target_path.exists() and target_path.stat().st_size > 0:
        return "skipped", file_name, 0, ""

    request = Request(file_url, headers={"User-Agent": "Mozilla/5.0"})

    try:
        with urlopen(request, timeout=timeout) as response, open(temp_path, "wb") as out_f:
            shutil.copyfileobj(response, out_f, length=1024 * 1024)

        temp_path.replace(target_path)
        file_size = target_path.stat().st_size
        return "downloaded", file_name, file_size, ""

    except HTTPError as exc:
        if temp_path.exists():
            temp_path.unlink()

        if exc.code == 404:
            return "missing", file_name, 0, ""

        return "error", file_name, 0, f"HTTP error {exc.code}"

    except URLError as exc:
        if temp_path.exists():
            temp_path.unlink()
        return "error", file_name, 0, f"Network error: {exc}"

    except Exception as exc:
        if temp_path.exists():
            temp_path.unlink()
        return "error", file_name, 0, f"Unexpected error: {exc}"


def iter_wet_file_urls(wet_paths_path: Path, download_web: str):
    if not wet_paths_path.exists():
        raise FileNotFoundError(f"wet paths file not found: {wet_paths_path}")

    download_web = download_web.rstrip("/")
    with gzip.open(wet_paths_path, "rt", encoding="utf-8") as manifest_file:
        for manifest_path in manifest_file:
            manifest_path = manifest_path.strip()
            if manifest_path:
                yield f"{download_web}/{manifest_path}"


def get_remote_file_size(file_url: str, timeout: int):
    request = Request(file_url, headers={"User-Agent": "Mozilla/5.0"}, method="HEAD")
    try:
        with urlopen(request, timeout=timeout) as response:
            content_length = response.headers.get("Content-Length")
            if content_length is None:
                return None
            return int(content_length)
    except (HTTPError, URLError, ValueError):
        return None


def report_download_result(status: str, file_name: str, message: str) -> tuple[int, int, int, int]:
    if status == "downloaded":
        print(f"Downloaded: {file_name}")
        return 1, 0, 0, 0

    if status == "skipped":
        print(f"Skip existing: {file_name}")
        return 0, 1, 0, 0

    if status == "missing":
        print(f"Missing: {file_name}")
        return 0, 0, 1, 0

    print(f"{message} for {file_name}")
    return 0, 0, 0, 1


def download_commoncrawl_from_wet_paths(
    data_directory_path: Path,
    wet_paths_path: Path,
    download_web: str = "https://data.commoncrawl.org",
    output_subdir: str = "CC_data",
    max_total_gb: float = 220.0,
    timeout: int = 120,
    max_workers: int = 10,
) -> None:
    output_dir = data_directory_path / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    max_total_bytes = int(max_total_gb * 1024**3)
    current_size_bytes = get_directory_size_bytes(output_dir)

    downloaded_count = 0
    skipped_count = 0
    missing_count = 0
    error_count = 0

    manifest_urls = iter_wet_file_urls(wet_paths_path, download_web)

    def schedule_downloads(
        executor: ThreadPoolExecutor,
        in_flight: dict,
        active_targets: set[Path],
        pending_known_bytes: int,
    ) -> tuple[int, bool]:
        stopped_for_size = False

        while len(in_flight) < max_workers:
            try:
                file_url = next(manifest_urls)
            except StopIteration:
                break

            target_path = output_dir / Path(file_url).name

            if target_path in active_targets:
                print(f"Skip in progress: {target_path.name}")
                continue

            if target_path.exists() and target_path.stat().st_size > 0:
                print(f"Skip existing: {target_path.name}")
                nonlocal skipped_count
                skipped_count += 1
                continue

            remote_file_size = get_remote_file_size(file_url, timeout)
            projected_size = current_size_bytes + pending_known_bytes

            if remote_file_size is not None and projected_size + remote_file_size > max_total_bytes:
                print(f"Stopped before downloading {target_path.name}: it would exceed {max_total_gb} GB")
                stopped_for_size = True
                break

            future = executor.submit(download_one_file, file_url, target_path, timeout)
            in_flight[future] = (target_path, remote_file_size)
            active_targets.add(target_path)

            if remote_file_size is not None:
                pending_known_bytes += remote_file_size

        return pending_known_bytes, stopped_for_size

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        if current_size_bytes >= max_total_bytes:
            print(f"Stopped: {output_dir} is already >= {max_total_gb} GB")
        else:
            in_flight = {}
            active_targets: set[Path] = set()
            pending_known_bytes = 0
            stop_submitting = False

            pending_known_bytes, stop_submitting = schedule_downloads(
                executor,
                in_flight,
                active_targets,
                pending_known_bytes,
            )

            while in_flight:
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)

                for future in done:
                    target_path, remote_file_size = in_flight.pop(future)
                    active_targets.discard(target_path)

                    if remote_file_size is not None:
                        pending_known_bytes -= remote_file_size

                    status, file_name, file_size, message = future.result()
                    if status == "downloaded":
                        current_size_bytes += file_size

                    downloaded_delta, skipped_delta, missing_delta, error_delta = report_download_result(
                        status,
                        file_name,
                        message,
                    )
                    downloaded_count += downloaded_delta
                    skipped_count += skipped_delta
                    missing_count += missing_delta
                    error_count += error_delta

                if current_size_bytes >= max_total_bytes:
                    print(f"Stopped: {output_dir} has reached {max_total_gb} GB")
                    stop_submitting = True

                if not stop_submitting:
                    pending_known_bytes, stop_submitting = schedule_downloads(
                        executor,
                        in_flight,
                        active_targets,
                        pending_known_bytes,
                    )

    final_size_gb = get_directory_size_bytes(output_dir) / (1024**3)
    print()
    print(f"Downloaded files: {downloaded_count}")
    print(f"Skipped existing files: {skipped_count}")
    print(f"Missing files: {missing_count}")
    print(f"Errored files: {error_count}")
    print(f"Current size under {output_dir}: {final_size_gb:.2f} GB")

data_directory_path = Path("/mnt/e/Data/cs336_data/Assignment4/")
wet_path = Path("/mnt/e/Data/cs336_data/Assignment4/wet.paths.gz")
download_web = "https://data.commoncrawl.org"

download_commoncrawl_from_wet_paths(
    data_directory_path=data_directory_path,
    wet_paths_path=wet_path,
    download_web=download_web,
    output_subdir="CC_data",
    max_total_gb=390.0,
    max_workers=5,
)