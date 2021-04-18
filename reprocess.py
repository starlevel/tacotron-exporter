# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


"""
A helper script that reprocesses data for training Tacotron models.
"""

import argparse
import csv
import functools
import random
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures._base import as_completed
from concurrent.futures.thread import ThreadPoolExecutor
from io import StringIO
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import re
else:
    import regex as re

import string
from pathlib import Path

try:
    import winreg
except ModuleNotFoundError:
    winreg = None

ALLOWED = set(string.ascii_letters + string.digits) | {".", "'"}
MATCHER = re.compile(r"(?:data/sound/voice/.*\.esm/)(.*)")
OUTPUT_PREFIX = "/content/drive/MyDrive/"
CSV_FIELD_NAMES = ("original_path", "voice_type", "response_text")


# These are all various utility functions
def _find_external(
    tool: str,
    filename: str,
) -> Path:
    """
    Tries to find an external tool.
    """
    path = shutil.which(filename)
    if path:
        return Path(path)

    # check our CWD (usual case), or in case the file is being overwritten is being overridden
    external_cwd = Path("./external")
    if external_cwd.exists():
        fullpath = external_cwd / tool / filename
        if fullpath.exists():
            return fullpath

    script_dir = Path(sys.modules["__main__"].__file__).parent
    fullpath = script_dir / tool / filename
    if fullpath.exists():
        return fullpath

    raise FileNotFoundError(
        f"Cannot find required exe {filename}. "
        f"Put it in your PATH or in <cwd>/external/{tool}"
    )


def find_skyrimse_path() -> Optional[Path]:
    """
    Finds the Skyrim SE install path.
    """
    if not winreg:
        return None

    with winreg.ConnectRegistry(None, winreg.HKEY_LOCAL_MACHINE) as registry:
        try:
            key = winreg.OpenKey(
                registry,
                r"SOFTWARE\Wow6432Node\Bethesda Softworks\Skyrim Special Edition",
            )
        except FileNotFoundError:
            return None

        with key:
            return Path(winreg.QueryValueEx(key, "Installed Path")[0])


def _find_directx_sdk() -> Optional[str]:
    """
    Finds the installed location of the DirectX SDK.
    """
    if not winreg:
        return None

    with winreg.ConnectRegistry(None, winreg.HKEY_LOCAL_MACHINE) as registry:
        try:
            key = winreg.OpenKey(
                registry,
                r"SOFTWARE\WOW6432Node\Microsoft\DirectX\Microsoft DirectX SDK (June 2010)",
            )
        except FileNotFoundError:
            return None

        with key:
            return Path(winreg.QueryValueEx(key, "InstallPath")[0])


def find_xwmaencode() -> Path:
    """
    Finds the xWMAEncode tool.
    """
    reg_path = _find_directx_sdk()
    if reg_path:
        root = Path(reg_path)
        return root / "utilities" / "bin" / "x86" / "xWMAEncode.exe"

    # check the PATH
    path = shutil.which("bsab")
    if path:
        return Path(path)

    # check our CWD (usual case)
    cwd_maybe = Path.cwd() / "xWMAEncode.exe"
    if cwd_maybe.exists():
        return cwd_maybe

    # finally just find it from our `__main__` location
    script_dir = Path(sys.modules["__main__"].__file__).parent
    fullpath = script_dir / "xWMAEncode.exe"
    if fullpath.exists():
        return fullpath

    raise FileNotFoundError(
        "Cannot find xWMAEncode exe. "
        "Put it in your PATH, the current working directory, or install the DirectX Legacy SDK"
    )


# These are all the actual command impls and helpers for those.
def produce_cleaned_file(input_file: Path, output_file: Path):
    """
    Produces a cleaned dialogue export file.
    """
    total = 0
    cut_total = 0

    with input_file.open(mode="r", newline="") as f1, output_file.open(
        mode="w", newline=""
    ) as f2:
        parser = csv.DictReader(f1, delimiter="\t")
        writer = csv.DictWriter(
            f2,
            fieldnames=CSV_FIELD_NAMES,
        )
        writer.writeheader()

        for row in parser:
            full_path = row["FULL PATH"]
            response = row["RESPONSE TEXT"]
            voice_type = row["VOICE TYPE"]
            response = response.strip()
            total += 1

            # no empty rows!
            if not response:
                continue

            # this strips out most of the ones with brackets and such
            if response[0] not in ALLOWED:
                continue

            cut_total += 1

            original_path = full_path
            # my own manual step, replace all \\ with /, so its easier to work with
            full_path = full_path.replace("\\", "/")

            # these steps are all according to the guide
            # 1) change XWM to WAV
            full_path = full_path.replace(".xwm", ".wav")
            # 2) lowercase everything
            full_path = full_path.lower()
            response = response.lower()
            # 3) reprefix if needed
            matched: str = MATCHER.findall(full_path)[0]
            writer.writerow(
                {
                    "original_path": original_path.replace("Data\\", ""),
                    "voice_type": voice_type,
                    "response_text": response,
                }
            )

    print(total, cut_total)


def list_voices(master_file: Path):
    """
    Lists the available voices for training.
    """
    counter = Counter()
    with master_file.open(mode="r", newline="") as f:
        reader = csv.DictReader(f)
        for line in reader:
            voice_type = line["voice_type"]
            counter[voice_type] += 1

    print(f"{'VOICE':<30} LINES")
    for (name, count) in counter.most_common():
        print(f"{name:<30} {count}")


## XX: the `-f` flag for bsab.exe doesn't seem to work...
## When possible, add proper output filtering.


def extract_voices(
    bsab_path: Path,
    input_file: Path,
    work_dir: Path,
):
    """
    Extracts voices from the Skyrim SE voices BSA.

    :param bsab_path: The path for ``bsab.exe``
    :param input_file: The ``Skyrim - Voices0_en.bsa`` file to use.
    :param work_dir: The directory to extract files into.
    """
    output_dir = work_dir / "extracted"
    output_dir.mkdir(exist_ok=True)

    subprocess.check_call((bsab_path, "-e", input_file, output_dir))


def remove_problematic_lines(
    infile: Path, outfile: Path, work_dir: Path, verbose: bool = False
) -> int:
    """
    Removes lines that have no corresponding voice file.
    """
    count = 0
    extracted = work_dir / "extracted"

    # The most well-known one is the really badly acted "Dovahkiin? Noooo" added back by USSEP.
    with infile.open(mode="r", newline="") as f1, outfile.open(
        mode="w", newline=""
    ) as f2:
        reader = csv.DictReader(f1)
        writer = csv.DictWriter(f2, fieldnames=CSV_FIELD_NAMES)
        writer.writeheader()
        for line in reader:
            file_path = extracted / line["original_path"].lower()
            file_path = file_path.with_suffix(".fuz")
            if not file_path.exists():
                if verbose:
                    print(f"File is in CSV but doesn't exist: {file_path}")
                count += 1
            else:
                writer.writerow(line)

    return count


def extract_fuz(input_file: Path, output_file: Path):
    """
    Extracts a single FUZ file into an XWM file.
    """
    # The FUZ file format is (seemingly) completely undocumented.
    # At least, I googled for a while and found NOTHING, except for references to other tools which
    # knew how to deal with the FUZ file.
    # Of course, these tools are all closed source! Because this community would rather die than
    # have their modifications be used Incorrectly.

    # A FUZ file is just a LIP file concatenated onto an XWM file with a header.
    # The header starts with the magic number FUZE.
    # Then a 4-byte number (seemingly always 0x1 in little endian? perhaps a version flag)
    # Then a 4-byte LIP file size (in little endian).
    # Presumably the LIP file follows for <size> bytes, then the XWM file is concatenated onto the
    # end.

    # We simply read the size, skip that many bytes, ensure we read off the RIFF header, and
    # write out the XWM file to the destination.

    with input_file.open(mode="rb") as f, output_file.open(mode="wb") as f2:
        magic_number = f.read(4)
        if magic_number != b"FUZE":
            raise ValueError(f"{input_file} does not seem to be a FUZ file")

        f.read(4)  # skip to RIFF
        size = int.from_bytes(f.read(4), byteorder="little", signed=False)
        f.seek(size, 1)

        magic = f.read(4)
        if magic != b"RIFF":
            raise ValueError(f"Got {magic} instead of a RIFF header in {input_file}")

        f2.write(b"RIFF")
        shutil.copyfileobj(f, f2)


def extract_all_fuz(cleaned_output: Path, work_dir: Path):
    """
    Extracts FUZ files into XWM files.
    """
    fuz_dir = work_dir / "extracted"
    xwa_dir = work_dir / "wav"

    # This is entirely I/O based so it's a good candidate to run it in parallel.
    candidates = []

    with cleaned_output.open(mode="r", newline="") as f:
        reader = csv.DictReader(f)
        for line in reader:
            infile = fuz_dir / line["original_path"].lower()
            infile = infile.with_suffix(".fuz")
            outfile = xwa_dir / line["original_path"].lower()
            outfile = outfile.with_suffix(".xwm")
            outfile.parent.mkdir(parents=True, exist_ok=True)
            candidates.append((infile, outfile))

    count = 0

    with ThreadPoolExecutor() as exe:
        futures = [exe.submit(lambda tp: extract_fuz(*tp), i) for i in candidates]
        for i in as_completed(futures):
            count += 1

            if count % 100 == 0:
                print(f"Extracted {count} files...")


def process_xwm(
    xwm_file: str, output_file: str, xwma_path: str, sox_path: str, ffmpeg_path: str
):
    """
    Processes an XWM file, turning it into a WAV file and applying the correct weaks.
    """
    # TODO: We could totally do this with a direct binding.
    # But that's a job for another day.

    # Semantic highlighting works wonders here!
    output_1 = output_file
    output_2 = output_file + "1.wav"

    # Step 1) Convert it to a WAV file.
    subprocess.check_call([xwma_path, xwm_file, output_file])
    # Step 2) Fix file using SoX.
    # It's not possible to write in place, so we swap between a _1 file and the original file
    # each time.
    # 2a) Trim silence at both ends. The doc does this an additional time, making the audio
    #     come out backwards. (I assume this is wrong.)
    args = "silence 1 0.1 1% reverse silence 1 0.1 1% reverse".split()
    subprocess.check_call([sox_path, output_1, output_2, *args])
    # 2b) Add silence back at the end.
    args = "pad 0 0.1".split()
    subprocess.check_call([sox_path, output_2, output_1, *args])
    # 2c) Downsample. This might be able to be done with ffmpeg, but I'm not sure.
    args = "rate -v 22050".split()
    subprocess.check_call([sox_path, output_1, output_2, *args])
    # 3) Mono-ify. The hide banner and loglevel params shut FFmpeg up.
    subprocess.check_call(
        [
            ffmpeg_path,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            output_2,
            "-ac",
            "1",
            output_1,
        ]
    )


# This is *the* killer function. It might well be better to run this on a WSL 2 share than raw NTFS
# because of how absolutely dogshit slow Windows FS perf is.
def generate_training_directory(cleaned_output: Path, work_dir: Path, voice_type: str):
    """
    Converts all of the XWM files to WAV files, and post-processes them. Then, generates the
    appropriate training CSV files.
    """
    # The singlethreaded XWM->WAV operation, in a very naiive manner, takes probably 20 minutes on
    # my box. That's not even including the post-processing steps.
    # We spawn a lot of threads to do this, but even then it's gonna take forever. I use 64 threads
    # because these are just dumb subprocess threads, but I haven't really benchmarked the optimum
    # thread count.

    # first we need our executables
    xwma_path = find_xwmaencode()
    sox_path = _find_external("sox", "sox.exe")
    ffmpeg_path = _find_external("ffmpeg", "ffmpeg.exe")

    xwm_dir = work_dir / "xwm"
    training_dir = work_dir / "training" / voice_type
    wav_output_dir = training_dir / "wavs"
    wav_output_dir.mkdir(parents=True, exist_ok=True)

    partials = []
    lines = []

    with cleaned_output.open(mode="r", newline="") as f:
        reader = csv.DictReader(f)
        for line in reader:
            if line["voice_type"] != voice_type:
                continue

            lines.append(line)

            infile = xwm_dir / line["original_path"].lower()
            infile = infile.with_suffix(".xwm")
            outfile = wav_output_dir / infile.name
            outfile = outfile.with_suffix(".wav")

            # this will save a godly amount of time on subsequent runs
            if not outfile.exists():
                fn = functools.partial(
                    process_xwm,
                    str(infile),
                    str(outfile),
                    str(xwma_path),
                    str(sox_path),
                    str(ffmpeg_path),
                )
                partials.append(fn)

    count = 0

    print(
        f"Need to process {len(partials)} lines. This is going to take a very long time."
    )
    with ThreadPoolExecutor() as exe:
        futures = [exe.submit(fn) for fn in partials]
        for i in as_completed(futures):
            count += 1

            if count % 10 == 0:
                print(f"Extracted {count} files...")

    # once that's done we can generate the _training.txt and _validation.txt
    def generate_file(picked_lines: list) -> StringIO:
        buf = StringIO()
        for line in picked_lines:
            buf.write(OUTPUT_PREFIX)
            buf.write(line["cleaned_path"])
            buf.write("|")
            buf.write(line["response_text"])
            buf.write("\n")

        buf.seek(0)
        return buf

    training_txt = training_dir / f"{voice_type.lower()}_training.txt"
    validation_txt = training_dir / f"{voice_type.lower()}_validation.txt"
    with training_txt.open(mode="w") as f:
        shutil.copyfileobj(generate_file(lines), f)

    sample_size = min(50, len(lines))
    validation_lines = random.sample(lines, k=sample_size)
    with validation_txt.open(mode="w") as f:
        shutil.copyfileobj(generate_file(validation_lines), f)


def with_duration(fn, message_prefix: str):
    """
    Runs a command with a duration check.
    """
    before = time.monotonic()
    result = fn()
    after = time.monotonic()
    print(message_prefix + f" ({after - before:.3f}).")
    return result


def main():
    """
    Main entrypoint.
    """

    parser = argparse.ArgumentParser(
        description="Processes Skyrim-related files for Tacotron training."
    )
    parser.add_argument(
        "-w",
        "--work-dir",
        help="The working directory for all created files.",
        type=Path,
        default=Path.cwd() / "out",
    )
    parser.add_argument(
        "-c",
        "--cleaned-output",
        default="cleaned_output.csv",
        help="The cleaned_output.csv file to use.",
    )

    parsers = parser.add_subparsers(dest="subcommand")

    # == Reprocessing == #
    reprocess_export = parsers.add_parser(
        "reprocess-export",
        help="Reprocesses the dialogueExport.txt file for later usage",
    )
    reprocess_export.add_argument(
        "input", help="The dialogueExport.txt generated from the CK", nargs="?"
    )

    # == Listing == #
    list_cmd = parsers.add_parser("list-voices", help="Lists all available voice types")

    # == Extracting BSA == #
    extract_cmd = parsers.add_parser(
        "extract-voices", help="Extracts all voice files from the BSA"
    )
    extract_cmd.add_argument(
        "-i",
        "--input-bsa",
        help="The input BSA file. Autodetected when possible.",
        default=None,
    )
    extract_cmd.add_argument(
        "--bsab-path",
        help="The path to the BSA Explorer CLI exe (bsab.exe). Autodetected when possible.",
    )

    # == Extracting XWM == #
    extract_xwm_cmd = parsers.add_parser(
        "extract-fuz", help="Extracts XWM files from the FUZ files"
    )

    generate_training_cmd = parsers.add_parser(
        "generate-training-data",
        help="Generates training data for the specified voice type",
    )
    generate_training_cmd.add_argument("voice_type", help="The voice type to generate")

    args = parser.parse_args()

    work_dir: Path = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    # both of these don't need the master file present
    if args.subcommand == "reprocess-export":
        if args.input is None:
            se_location = find_skyrimse_path()
            if se_location is None:
                parser.error(
                    "Failed to locate dialogueExport.txt automatically. "
                    "Specify its full path manually."
                )
            input_file = se_location / "dialogueExport.txt"
        else:
            input_file = Path(args.input)

        if not input_file.exists():
            parser.error(f"Input path {input_file} does not exist.")

        output_file = Path(args.cleaned_output)
        fn = functools.partial(produce_cleaned_file, input_file, output_file)
        with_duration(fn, f"Written cleaned output file to {output_file}")
        return

    # these commands all rely on the cleaned output CSV
    cleaned_output = Path(args.cleaned_output)
    if not cleaned_output.exists():
        parser.error(
            f"Could not find {cleaned_output}. Run 'reprocess-export' to generate it."
        )

    if args.subcommand == "extract-voices":
        input_bsa = args.input_bsa
        if input_bsa is None:
            skyrim_path = find_skyrimse_path()
            if skyrim_path is None:
                print(
                    "Failed to find Skyrim SE path from the Windows Registry. Specify the path "
                    "manually."
                )
                return

            input_bsa = skyrim_path / "Data" / "Skyrim - Voices_en0.bsa"

        bsab = args.bsab_path
        if bsab is None:
            bsab = _find_external("bsa_browser", "bsab.exe")

        fn = functools.partial(extract_voices, bsab, input_bsa, work_dir)
        with_duration(fn, "Extracted all voice files.")
        print(
            f"Some voice lines are missing from the BSA. Removing them from the CSV..."
        )
        backup = cleaned_output.with_suffix(".bak")
        shutil.move(cleaned_output, cleaned_output.with_suffix(".bak"))

        before = time.monotonic()
        cnt = remove_problematic_lines(backup, cleaned_output, work_dir, verbose=True)
        after = time.monotonic()
        print(f"Removed {cnt} problematic lines. ({after - before:.3f}s)")

        return

    elif args.subcommand == "list-voices":
        return list_voices(cleaned_output)
    elif args.subcommand == "extract-fuz":
        fn = functools.partial(extract_all_fuz, cleaned_output, work_dir)
        return with_duration(fn, "Extracted all FUZ files")
    elif args.subcommand == "generate-training-data":
        fn = functools.partial(
            generate_training_directory, cleaned_output, work_dir, args.voice_type
        )
        return with_duration(
            fn, "Reprocessed all voice files and generated training lists"
        )


if __name__ == "__main__":
    main()
