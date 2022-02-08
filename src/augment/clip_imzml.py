
import argparse
import re
import collections
from typing import Iterator, Optional, Tuple, OrderedDict


def find_unique_accessors(
    text: str,
    *accessor_re: str
) -> OrderedDict[str, Optional[re.Match]]:

    if not accessor_re:
        return collections.OrderedDict({})

    re_lst = {
        pattern: re.compile(rf'<cvParam [^>]*accession="{pattern}"[^>]*>')
        for pattern in accessor_re
    }

    matches = {key: pattern.search(text) for key, pattern in re_lst.items()}

    def sort_key(match_item: Tuple[str, Optional[re.Match]]):
        if not match_item[1]:
            return -1
        return match_item[1].start(0)

    return collections.OrderedDict(sorted(matches.items(), key=sort_key))


def get_encoding(xml: str) -> str:
    # figure out encodings
    with open(xml, mode='rb') as source_file:
        line = next(source_file, '')
        if not line:
            raise ValueError(f"empty file {xml}")

        prefix = b'encoding="'
        idx_start = line.index(prefix) + len(prefix)

        idx_end = line.index(b'"', idx_start)

        encoding = line[idx_start:idx_end].decode('ASCII')

        return encoding


class Clipper:
    def __init__(self, clip_x: int, clip_y: int) -> None:
        self.idx = 0
        self.data = ''
        self.clip_x = clip_x
        self.clip_y = clip_y

        self.counted = 0

        if not isinstance(clip_x, int) or clip_x < 1:
            raise ValueError(f'{clip_x=} should be a positive integer')
        if not isinstance(clip_y, int) or clip_y < 1:
            raise ValueError(f'{clip_y=} should be a positive integer')

    def parse(self, source: str, dest: str):

        # open text file & load it in memory at once
        try:
            encoding = get_encoding(source)
            with open(source, mode='r', encoding=encoding) as source_file:
                self.data = source_file.read()
        except MemoryError as err:
            raise ValueError('XML file is too large') from err

        # remove spectra
        with open(dest, mode='w', encoding=encoding) as dest_file:
            for part in self.content():
                dest_file.write(part)

        # re-read data
        with open(dest, mode='r', encoding=encoding) as dest_file:
            self.data = dest_file.read()

        # write data with fixed counts
        self.idx = 0
        with open(dest, mode='w', encoding=encoding) as dest_file:
            for part in self.fix_count():
                dest_file.write(part)
            for part in self.iter_footer():
                dest_file.write(part)

    def _yield_until_match(self, match: re.Match) -> Iterator[str]:
        "make sure all text before match has been yielded"
        if not match:
            return
        low = match.start(0)
        if low != self.idx:
            yield self.data[self.idx:low]
            self.idx = low

    def fix_count(self) -> Iterator:

        # get spectrumList definition
        spectrum_list = re.search(r'<spectrumList [^>]*>', self.data)
        yield from self._yield_until_match(spectrum_list)

        count_kv = re.search(r'count="(\d+)"', spectrum_list[0])[0]

        self.idx = spectrum_list.end(0)
        yield spectrum_list[0].replace(count_kv, f'count="{self.counted}"')

    def iter_header(self) -> Iterator:
        """iterate and update header accessor regarding: max count pixel x
        (IMS:1000042), max count pixel y (IMS:1000043)
        Yields:
            Iterator: [description]
        """

        def update_width(match: re.Match) -> str:
            """
            - expects that self.idx == match.start(0)
            - update match[0] to set the width to *= self.repeat_x
            - yield the updated accessor
            - set self.idx == match.end(0)
            """
            attr = match[0]
            width_kv = re.search(r'value="(\d+)"', attr)[0]

            self.idx = match.end(0)
            return attr.replace(width_kv, f'value="{self.clip_x}"')

        def update_height(match: re.Match) -> str:
            """
            - expects that self.idx == match.start(0)
            - update match[0] to set the height to *= self.repeat_y
            - yield the updated accessor
            - set self.idx == match.end(0)
            """
            attr = match[0]
            height_kv = re.search(r'value="(\d+)"', attr)[0]

            self.idx = match.end(0)
            return attr.replace(height_kv, f'value="{self.clip_y}"')

        param_update = {
            'IMS:1000042' : update_width,
            'IMS:1000043' :  update_height,
        }

        matches = find_unique_accessors(
            self.data,
            *param_update.keys()
        )

        for match in matches.items():
            yield from self._yield_until_match(match[1])
            yield param_update[match[0]](match[1])

    def iter_spectra(self) -> Iterator:

        spectrum_start_re = re.compile(r'<spectrum ')
        spectrum_end_re = re.compile(r'</spectrum>')
        index_re = re.compile(r'index="(\d+)"')
        pos_x_re = re.compile(r'<cvParam [^>]*accession="IMS:1000050"[^>]*>')
        pos_y_re = re.compile(r'<cvParam [^>]*accession="IMS:1000051"[^>]*>')
        value_re = re.compile(r'value="(\d+)"')

        def yield_filtered_spectrum(original: str):
            # find idx
            index_kv = index_re.search(original)[0]
            old_idx = int(index_kv[7:-1])

            # find x, y
            x_attr = pos_x_re.search(original)[0]
            x_kv = value_re.search(x_attr)[0]
            old_x = int(x_kv[7:-1])

            y_attr = pos_y_re.search(original)[0]
            y_kv = value_re.search(y_attr)[0]
            old_y = int(y_kv[7:-1])

            if old_x > self.clip_x:
                return
            if old_y > self.clip_y:
                return

            spectrum = original.replace(index_kv, f'value="{self.counted}"')
            spectrum = spectrum.replace(
                f'spectrum={old_idx}',
                f'spectrum={self.counted}'
            )

            self.counted += 1

            yield spectrum

        while True:
            start = spectrum_start_re.search(self.data, pos=self.idx)
            end = spectrum_end_re.search(self.data, self.idx)

            if start is None:
                break

            yield from self._yield_until_match(start)

            # get spectrum
            spectrum = self.data[start.start(0):end.end(0)]

            yield from yield_filtered_spectrum(spectrum)

            self.idx = end.end(0)

    def iter_footer(self) -> Iterator:
        yield self.data[self.idx:]
        self.idx = -1

    def content(self) -> Iterator:
        yield from self.iter_header()
        yield from self.iter_spectra()
        yield from self.iter_footer()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('imzml_in', type=str)
    parser.add_argument('imzml_out', type=str)
    parser.add_argument('clip_x', type=int)
    parser.add_argument('clip_y', type=int)

    args = parser.parse_args()

    Clipper(args.clip_x, args.clip_y).parse(args.imzml_in, args.imzml_out)
