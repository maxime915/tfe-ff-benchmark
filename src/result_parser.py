"result_parser: analyze logs and present results"

import sys
import parse

if __name__ != "__main__":
    raise ImportError("module should not be imported")

if len(sys.argv) != 2:
    raise ValueError(
        f"expected a single file as argument, found {sys.argv[1:]}")

hdf5_results = {}
zarr_compression_results = {}
zarr_results = {}

with open(sys.argv[1], mode='r') as result_file:
    lines = result_file.readlines()
    # skip nohup
    if lines[0] == 'nohup: ignoring input\n':
        lines = lines[1:]

    # N of HDF5 benchmark
    while lines[0].startswith('Tile3DHDF5Benchmark'):
        number, repeat = parse.parse(
            'Tile3DHDF5Benchmark (duration averaged on {:d} iterations, repeated {:d} times.)\n', lines[0])

        # trick to allow unpacking: parse '\t' in a dummy variable (also used later)
        _, *results = parse.parse(
            '{}[' + ', '.join(repeat * ['{:e}']) + ']\n', lines[1])

        min_val, max_val = parse.parse(
            '\tresults.min() = {:e} s | results.max() = {:e} s\n', lines[2])

        mean_val, std_val = parse.parse(
            '\tmean = {:e} s ± std = {:e} s\n', lines[3])

        hdf5_path, *tiles = parse.parse(
            'HDF5 Companion: {}, using tile = [{:d}, {:d}, {:d}]\n', lines[4])

        shape, chunks, compression = parse.parse(
            '\tshape = {}, chunks = {}, compression = {}\n', lines[5])
        # chunks is None all the time, otherwise it could be useful to evaluate it

        lines = lines[7:]

        hdf5_results[tuple(tiles)] = (min_val, max_val, mean_val, std_val, results)

    while lines:
        # conversion
        _, _, *chunks, order, compressor = parse.parse(
            "converting {} to {} (chunk=[{:d}, {:d}, {:d}], order='{}', compressor={})\n", lines[0])

        # this may break if the repeat value is altered
        _, *results = parse.parse('{}[{:e}, {:e}, {:e}]\n', lines[1])

        min_val, max_val = parse.parse(
            '\tresults.min() = {:e} s | results.max() = {:e} s\n', lines[2])

        mean_val, std_val = parse.parse(
            '\tmean = {:e} s ± std = {:e} s\n', lines[3])

        lines = lines[5:]

        zarr_compression_results[(tuple(chunks), order, compressor)] = (min_val, max_val, mean_val, std_val, results)

        # N of Zarr benchmark
        for _ in range(len(hdf5_results)):  # iterates the same amount of time

            number, repeat = parse.parse(
                'Tile3DZarrBenchmark (duration averaged on {:d} iterations, repeated {:d} times.)\n', lines[0])

            _, *results = parse.parse(
                '{}[' + ', '.join(repeat * ['{:e}']) + ']\n', lines[1])

            min_val, max_val = parse.parse(
                '\tresults.min() = {:e} s | results.max() = {:e} s\n', lines[2])

            mean_val, std_val = parse.parse(
                '\tmean = {:e} s ± std = {:e} s\n', lines[3])

            hdf5_path, *tiles = parse.parse(
                'Zarr file: {}, using tile = [{:d}, {:d}, {:d}]\n', lines[4])

            shape, *_, _, filters = parse.parse(
                '\tshape = {}, chunks = ({:d}, {:d}, {:d}), compressor = {}, filters = {}\n', lines[5])

            _, _, size_full, _, size_disk = parse.parse(
                '\torder = {}, size (mem) = {} ({}), size (disk) {} ({})\n', lines[6])

            lines = lines[8:]

            zarr_results[(tuple(tiles), tuple(chunks), order, compressor)] = (min_val, max_val, mean_val, std_val, results)
        
        # add size information to the zarr_compression_results
        zarr_compression_results[(tuple(chunks), order, compressor)] += (size_full, size_disk)

# everything is parsed, time to present data !

# tile effect on HDF5 from the HMS module
# TODO

# chunk effect on Zarr compression (one curve per memory order & compressor)
# TODO

# tile effect on Zarr (one curve per memory order & compressor)
# TODO 2-3 window for a few chunk value (lowest, ?, highest)

# tile effect on Zarr (one curve per chunk value)