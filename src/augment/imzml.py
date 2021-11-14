"""augment.imzml: augment imzML image without changing the ibd file


Limitations
  - XML is a very broad dataformat, the comparison is done based on text rather
      than parsed value which means not all files will be supported
  - source must be an image in continuous mode
  - image may be repeated an integer amount of time in either direction
"""

import argparse
import re
from typing import Callable, List

import parse


def augment(source: str, destination: str, to_processed: bool,
            repeat_x: int = 1, repeat_y: int = 1) -> None:
    """augment: create an augmented image from source

    Args:
        source (str): path to the source continuous ImzML image
        destination (str): path to the destination .imzML file (.ibd is the same)
        to_processed (bool): create a processed image instead of a continuous one
        repeat_x (int, optional): amount of time the image is repeated in the X axis. 1 means the image is left as is. Defaults to 1.
        repeat_y (int, optional): amount of time the image is repeated in the Y axis. 1 means the image is left as is. Defaults to 1.

    Raises:
        ValueError: if the argument aren't correct
    """

    if repeat_x <= 0 or repeat_y <= 0:
        raise ValueError('repeat values must be positive integers')

    update_fn: Callable[[str], str] = lambda s: s
    shape: List[int] = [0, 0]

    # functional approach : update_fn is the current mapper, it is fed each line
    #   and returns what to write to the output file (may be empty)

    def copy_footer(line: str) -> str:
        "copies everything else, as is"
        # this function is probably not necessary, as copy_spectrum copies what
        # it doesn't recognize
        return line

    def copy_spectrum(line: str) -> str:
        "copies the spectrum of the imzML files with an updated index"
        nonlocal update_fn

        # end of spectrum: process pending lines
        if '      </spectrum>\n' == line:
            # raw str
            copy_spectrum.pending.append(line)
            spectrum_str = ''.join(copy_spectrum.pending)
            copy_spectrum.pending.clear()

            # list of spectrums to return
            processed: List[str] = []

            # read position
            try:
                pos_x = int(re.findall('name="position x" value="(\d+)"', spectrum_str)[0])
                pos_y = int(re.findall('name="position y" value="(\d+)"', spectrum_str)[0])
            except IndexError as e:
                print(f'Unable to parse position in {spectrum_str=}')
                raise e

            for i in range(repeat_x):
                for j in range(repeat_y):
                    # replace index
                    spectrum = re.sub(r'id="spectrum=\d" index="\d" spotID="\d"',
                                      f'id="spectrum={copy_spectrum.idx}" index="{copy_spectrum.idx}" spotID="{copy_spectrum.idx}"',
                                      spectrum_str)

                    # replace x, y
                    spectrum = re.sub(r'name="position x" value="\d"',
                                      f'name="position x" value="{i * shape[0] + pos_x}"',
                                      spectrum)
                    spectrum = re.sub(r'name="position y" value="\d"',
                                      f'name="position y" value="{j * shape[1] + pos_y}"',
                                      spectrum)

                    # other value are ignored

                    # add to processed list
                    processed.append(spectrum)

                    # update idx
                    copy_spectrum.idx += 1

            return ''.join(processed)

        # end of list, change the update_fn (pending should be empty)
        if '    </spectrumList>\n' == line:
            assert(len(copy_spectrum.pending) == 0)
            update_fn = copy_footer
            return line

        # add line to pending list, don't write anything
        copy_spectrum.pending.append(line)
        return ''

    copy_spectrum.idx = 0  # index of the next spectrum to write
    copy_spectrum.pending = []  # pending lines to process

    def copy_header(line: str) -> str:
        nonlocal update_fn, shape

        # update binary mode
        if to_processed and '      <cvParam accession="IMS:1000030" cvRef="IMS" name="continuous"/>\n' == line:
            return '      <cvParam accession="IMS:1000031" cvRef="IMS" name="processed"/>\n'

        # update width
        width = parse.parse('      <cvParam accession="IMS:1000042" cvRef="IMS" name="max count of pixels x" value="{:d}"/>\n', line)
        if width is not None:
            shape[0] = width[0]
            return f'      <cvParam accession="IMS:1000042" cvRef="IMS" name="max count of pixels x" value="{repeat_x * width[0]}"/>\n'

        # update height
        height = parse.parse('      <cvParam accession="IMS:1000043" cvRef="IMS" name="max count of pixels y" value="{:d}"/>\n', line)
        if height is not None:
            shape[1] = height[0]
            return f'      <cvParam accession="IMS:1000043" cvRef="IMS" name="max count of pixels y" value="{repeat_y * height[0]}"/>\n'

        # update spectrum count
        count = parse.parse('    <spectrumList count="{:d}" defaultDataProcessingRef="dataProcessing0">\n', line)
        if count is not None:
            update_fn = copy_spectrum
            return f'    <spectrumList count="{repeat_x * repeat_y * count[0]}" defaultDataProcessingRef="dataProcessing0">\n'

        # otherwise repeat info
        return line

    update_fn = copy_header

    with open(destination, 'w') as dest:
        with open(source, 'r') as src:
            for line in src:
                dest.write(update_fn(line))

    assert(len(copy_spectrum.pending) == 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('imzml_in', type=str)
    parser.add_argument('imzml_out', type=str)
    parser.add_argument('--repeat-x', type=int, default=1, dest='repeat_x')
    parser.add_argument('--repeat-y', type=int, default=1, dest='repeat_y')
    parser.add_argument('--make-processed',
                        dest='processed', action='store_true')
    parser.set_defaults(processed=False)

    args = parser.parse_args()

    augment(args.imzml_in, args.imzml_out,
            args.processed, args.repeat_x, args.repeat_y)
