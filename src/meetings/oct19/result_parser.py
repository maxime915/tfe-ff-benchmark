"result_parser: analyze logs and present results"

import itertools
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import parse

if __name__ != "__main__":
    raise ImportError("module should not be imported")

if len(sys.argv) != 2:
    raise ValueError(
        f"expected a single file as argument, found {sys.argv[1:]}")

hdf5 = pd.DataFrame()
zarr_conv = pd.DataFrame()
zarr = pd.DataFrame()

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

        hdf5 = hdf5.append({'tile_x': tiles[0],
                            'tile_z': tiles[2],
                            'mean': mean_val,
                            'std': std_val},
                           ignore_index=True)

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

        zarr_conv_it = {
            'chunk_x': chunks[0],
            'chunk_z': chunks[2],
            'order': order,
            'compressor': compressor,
            'mean': mean_val,
            'std': std_val}

        # N of Zarr benchmark
        for _ in range(len(hdf5)):  # iterates the same amount of time

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

            zarr = zarr.append({'tile_x': tiles[0],
                                'tile_z': tiles[2],
                                'chunk_x': chunks[0],
                                'chunk_z': chunks[2],
                                'order': order,
                                'compressor': compressor,
                                'mean': mean_val,
                                'std': std_val},
                               ignore_index=True)

        # add size information to zarr_conv
        zarr_conv_it['size_full'] = size_full
        zarr_conv_it['size_disk'] = size_disk
        zarr_conv = zarr_conv.append(zarr_conv_it, ignore_index=True)

# tile effect on HDF5 from the HMS module: roughly linear
if True and False:
    plt.figure()
    plt.title('HDF5 access time (averaged on 200 iterations, repeated 5 times)')

    plt.plot(
        hdf5[hdf5['tile_z'] == -1]['tile_x'],
        hdf5[hdf5['tile_z'] == -1]['mean'],
        '-', marker='x', color='b', label=f'full tile',
    )
    plt.fill_between(
        hdf5[hdf5['tile_z'] == -1]['tile_x'],
        hdf5[hdf5['tile_z'] == -1]['mean'] -
        hdf5[hdf5['tile_z'] == -1]['std'],
        hdf5[hdf5['tile_z'] == -1]['mean'] +
        hdf5[hdf5['tile_z'] == -1]['std'],
        color='b', alpha=.1
    )

    plt.plot(
        hdf5[hdf5['tile_z'] == 1]['tile_x'],
        hdf5[hdf5['tile_z'] == 1]['mean'],
        '-', marker='x', color='g', label=f'flat tile',
    )
    plt.fill_between(
        hdf5[hdf5['tile_z'] == 1]['tile_x'],
        hdf5[hdf5['tile_z'] == 1]['mean'] -
        hdf5[hdf5['tile_z'] == 1]['std'],
        hdf5[hdf5['tile_z'] == 1]['mean'] +
        hdf5[hdf5['tile_z'] == 1]['std'],
        color='g', alpha=.1
    )

    plt.xticks(2**np.arange(7, 11))
    plt.xlabel('Tile width (pixel)')
    plt.ylabel('Access time (s)')
    plt.legend()
    plt.show()
    

# chunk effect on Zarr compression (one curve per memory order & compressor)
if True and False:
    # being 'flat' is slower, then having 'F' memory order then compression:
    # barely no effect

    fig, ax = plt.subplots()
    ax.set_title('Zarr conversion time (averaged on 2 iterations, repeated 3 times)')

    for compressor in zarr_conv['compressor'].unique():
        data_ = zarr_conv[zarr_conv.compressor.eq(compressor)]

        # order = C, deep chunk
        data = data_[data_.chunk_z.eq(-1) & data_.order.eq('C')]
        ax.plot(
            data['chunk_x'],
            data['mean'],
            ls='--',
            marker='x',
            label=f'deep chunks/C/{compressor}',
        )

        # order = F, deep chunk
        data = data_[data_.chunk_z.eq(-1) & data_.order.eq('F')]
        ax.plot(
            data['chunk_x'],
            data['mean'],
            ls='-',
            marker='x',
            label=f'deep chunks/F/{compressor}',
            color=ax.get_lines()[-1].get_color(),
        )

        # order = C, flat chunk
        data = data_[data_.chunk_z.eq(1) & data_.order.eq('C')]
        ax.plot(
            data['chunk_x'],
            data['mean'],
            ls=':',
            marker='x',
            label=f'flat chunks/C/{compressor}',
            color=ax.get_lines()[-1].get_color(),
        )

        # order = F, flat chunk
        data = data_[data_.chunk_z.eq(1) & data_.order.eq('F')]
        ax.plot(
            data['chunk_x'],
            data['mean'],
            ls='-.',
            marker=6,
            markevery=.1,
            label=f'flat chunks/F/{compressor}',
            color=ax.get_lines()[-1].get_color(),
        )

    ax.set_xticks(2**np.arange(7, 11))
    ax.set_xlabel('Chunk width (pixel)')
    ax.set_ylabel('Conversion time (s)')
    ax.legend()
    plt.show()


# WARNING: creates a lot of plots
if True and False:
    # having no compressor is faster
    #   larger chunks are more affected by it
    #   chunk depth ? no apparent relation...
    #   memory order ? no apparent relation...

    # deeper chunk are faster to access (% variable, but always true)

    # memory order has a
    #   - small impact for deep chunks
    #   - large impact for flat chunks
    for order in zarr.order.unique():
        for compressor in zarr.compressor.unique():
            for chunk_x in zarr.chunk_x.unique():
                for chunk_z in zarr.chunk_z.unique():
                    data = zarr[zarr.order.eq(order)
                        & zarr.compressor.eq(compressor)
                        & zarr.chunk_x.eq(chunk_x)
                        & zarr.chunk_z.eq(chunk_z)]

                    print(data)
                    break

                    if len(data) == 0:
                        continue
                    
                    data_flat = data[data.tile_z.eq(+1)]
                    data_deep = data[data.tile_z.eq(-1)]

                    title = f'figure_{compressor}_{chunk_x}_{chunk_z}_{order}.png'
                    fig = plt.figure()
                    plt.title(title)
                    
                    plt.plot(data_flat['tile_x'], data_flat['mean'], label='flat tile')
                    plt.plot(data_deep['tile_x'], data_deep['mean'], label='deep tile')
                    plt.legend()
                    plt.xticks(2**np.arange(7, 11))
                    plt.ylim(0.0, 3.0)

                    fig.savefig(title)
                    plt.close(fig)

def slope(x: pd.Series, y: pd.Series) -> float:
    "compte the slope of a simple linear regression y = **s** * x + p"

    x_centered = x - x.mean()
    y_centered = y - x.mean()
    return np.dot(x_centered, y_centered) / np.dot(x_centered, x_centered)

def slope_from_rows(rows: pd.DataFrame) -> float:
    return slope(rows.tile_x, rows['mean'])

if True and False:
    ## plot regression coefficient [tile width -> access time]
    # variate (memory order x compressor, tile depth, chunk depth)
    #   -> show how larger chunks are more affected by compression
    #   -> memory order shouldn't make too much of a difference
    fig, axes = plt.subplots(2, 2)
    axes = axes.flatten()
    fig.suptitle('Access time over tile width slope for different chunk sizes')

    for ax_index, (tile_z, chunk_z) in enumerate(itertools.product((-1, 1), repeat=2)):
        ax = axes[ax_index]
        data_ = zarr[zarr.tile_z.eq(tile_z) & zarr.chunk_z.eq(chunk_z)]
        for (order, compressor) in itertools.product(data_.order.unique(), data_.compressor.unique()):
            data = data_[data_.order.eq(order) & data_.compressor.eq(compressor)]

            ax.plot(
                data.chunk_x.unique(),
                [slope_from_rows(data[data.chunk_x.eq(t)]) for t in data.chunk_x.unique()],
                label=f'{order}/{compressor}',
            )

            name = lambda z: 'flat' if z == 1 else 'deep'
        
            ax.set_title(f'Settings: {name(chunk_z)} chunks, {name(tile_z)} tiles.')
            ax.set_xticks(2**np.arange(7, 7+len(data.chunk_x.unique())))
            ax.set_ylabel('Slope (s/tile pixel)')
            ax.set_xlabel('Chunk width (pixel)')
            ax.legend()
        
    plt.show()

    # on deep chunk, the compressor effect is larger than the memory order effect
    # on flat chunk, it is the opposite

    # why do flat chunk performances get worse wrt chunk width, even for flat tiles ????


if True and False:
    ## plot access time at tile width=512
    # variate (memory order x compressor, tile depth, chunk depth)
    #   -> show how larger chunks are more affected by compression
    #   -> memory order shouldn't make too much of a difference
    fig, axes = plt.subplots(2, 2)
    axes = axes.flatten()
    tile_x = 512
    fig.suptitle(f'Access time for a tile of size {tile_x} at different chunk sizes')

    for ax_index, (tile_z, chunk_z) in enumerate(itertools.product((-1, 1), repeat=2)):
        ax = axes[ax_index]
        data_ = zarr[zarr.tile_z.eq(tile_z) & zarr.tile_x.eq(tile_x) & zarr.chunk_z.eq(chunk_z)]
        for (order, compressor) in itertools.product(data_.order.unique(), data_.compressor.unique()):
            data = data_[data_.order.eq(order) & data_.compressor.eq(compressor)]

            ax.plot(
                data.chunk_x.unique(),
                data['mean'],
                label=f'{order}/{compressor}',
            )

            name = lambda z: 'flat' if z == 1 else 'deep'
        
            ax.set_title(f'Settings: {name(chunk_z)} chunks, {name(tile_z)} tiles.')
            ax.set_xticks(2**np.arange(7, 7+len(data.chunk_x.unique())))
            # ax.set_ylim(0.25, 1.5)
            ax.set_ylabel('Access time for a tile of 512px by 512px (s)')
            ax.set_xlabel('Chunk width (pixel)')
            ax.legend()
        
    plt.show()

    # on deep chunk, the compressor effect is larger than the memory order effect
    # on flat chunk, it is the opposite

    # why do flat chunk performances get worse wrt chunk width, even for flat tiles ????

def plot_all(axis, data: pd.DataFrame, label: str) -> None:
    reuse_color = False
    for compressor in zarr.compressor.unique():  # 2
        for order in zarr.order.unique():  # 2
            for chunk_z in zarr.chunk_z.unique():  # 2
                for chunk_x in zarr.chunk_x.unique():  # 3 or 4
                    for tile_z in zarr.tile_z.unique():
                        data_ = data[data.compressor.eq(compressor)
                            & data.order.eq(order)
                            & data.chunk_z.eq(chunk_z)
                            & data.chunk_x.eq(chunk_x)
                            & data.tile_z.eq(tile_z)]

                        kwargs = {'alpha': 0.5}
                        if reuse_color:
                            kwargs['color'] = axis.get_lines()[-1].get_color()
                        else:
                            kwargs['label'] = label
    
                        axis.plot(
                            data_.tile_x.unique(),
                            data_['mean'],
                            **kwargs,
                        )

                        reuse_color = True

if True and False:
    # rough comparison -> everything with two colors, low alphas
    truncated = zarr[zarr.tile_x.le(512) & zarr.chunk_x.le(512)]

    fig, axis = plt.subplots()
    for compressor in truncated.compressor.unique():
        plot_all(axis, truncated[truncated.compressor.eq(compressor)], compressor)
    axis.set_title(f'Comparison of different compressor options')
    axis.set_xticks(2**np.arange(7, 7+len(truncated.tile_x.unique())))
    axis.set_ylabel('Access time (s)')
    axis.set_xlabel('Tile width (pixel)')
    axis.legend()
    plt.show()

    fig, axis = plt.subplots()
    for order in truncated.order.unique():
        plot_all(axis, truncated[truncated.order.eq(order)], order)
    axis.set_title(f'Comparison of different order options')
    axis.set_xticks(2**np.arange(7, 7+len(truncated.tile_x.unique())))
    axis.set_ylabel('Access time (s)')
    axis.set_xlabel('Tile width (pixel)')
    axis.legend()
    plt.show()

    fig, axis = plt.subplots()
    for chunk_z in truncated.chunk_z.unique():
        plot_all(axis, truncated[truncated.chunk_z.eq(chunk_z)], chunk_z)
    axis.set_title(f'Comparison of different chunk_z options')
    axis.set_xticks(2**np.arange(7, 7+len(truncated.tile_x.unique())))
    axis.set_ylabel('Access time (s)')
    axis.set_xlabel('Tile width (pixel)')
    axis.legend()
    plt.show()

    fig, axis = plt.subplots()
    for chunk_x in truncated.chunk_x.unique():
        plot_all(axis, truncated[truncated.chunk_x.eq(chunk_x)], chunk_x)
    axis.set_title(f'Comparison of different chunk_x options')
    axis.set_xticks(2**np.arange(7, 7+len(truncated.tile_x.unique())))
    axis.set_ylabel('Access time (s)')
    axis.set_xlabel('Tile width (pixel)')
    axis.legend()
    plt.show()


if True:

    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(9, 4))
    
    for axis, depth in zip(axes, [1, -1]):
        tile_x = 1024 if depth == 1 else 512
        axis.set_title("Window depth: " + {1: "flat", -1: "full"}[depth])
        axis.set_xlabel("Tile width (pixels)")
        axis.set_xlabel('Tile width (pixel)')
        axis.set_xticks([128, 256, 512] + ([1024] if depth == 1 else []))

        h5_fdepth = hdf5[hdf5.tile_z == depth]
        z_fdepth = zarr[zarr.tile_z == depth]
        
        best_vals = z_fdepth[z_fdepth.tile_x == tile_x]
        best_z_idx = np.argsort(best_vals["mean"])
        best_z_idx = best_z_idx[:3]
    

        axis.plot(h5_fdepth.tile_x, h5_fdepth["mean"], label="HDF5")
        for z_idx in best_z_idx:
            assert z_idx < len(z_fdepth), f"{z_idx}, {len(z_fdepth)}"
            row = best_vals.iloc[z_idx]
            label = (
                f"Zarr: (c={row.compressor}, o={row.order} c=({row.chunk_x}, {row.chunk_z})"
            )
            selection = z_fdepth[
                z_fdepth.compressor.eq(row.compressor)
                & z_fdepth.order.eq(row.order)
                & z_fdepth.chunk_x.eq(row.chunk_x)
                & z_fdepth.chunk_z.eq(row.chunk_z)
            ]

            assert len(selection) == len(h5_fdepth), "inconsistent sizes"
            axis.plot(selection.tile_x, selection["mean"], label=label)
        
        axis.legend()

    fig.suptitle("HDF5 - Zarr comparison")
    fig.tight_layout()
    
    fig.savefig("hdf5_zarr_comparison.png")
