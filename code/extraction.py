import argparse
import json
import logging
import os
from datetime import datetime as dt
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Tuple, Union

import cv2
import h5py
import imageio_ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import skimage
import sparse
import suite2p
from aind_data_schema.core.processing import DataProcess, ProcessName
from aind_ophys_utils.array_utils import downsample_array
from aind_ophys_utils.summary_images import max_corr_image


def get_r_from_min_mi(raw_trace, neuropil_trace, resolution=0.01, r_test_range=[0, 2]):
    """
    Get the r value that minimizes the mutual information between
    the corrected trace and the neuropil trace.

    Parameters
    ----------
    raw_trace : np.ndarray
        1D array of raw trace values.
    neuropil_trace : np.ndarray
        1D array of neuropil trace values.
    resolution : float
        Resolution of r values to test.
    r_test_range : list of float
        List of two floats representing the inclusive range of r values to test.

    Returns
    -------
    r_best : float
        r value that minimizes the mutual information between
        the corrected trace and the neuropil trace.
    mi_iters : np.ndarray
        1D array of mutual information values for each r value tested.
    r_iters : np.ndarray
        1D array of r values tested.
    """
    r_iters = np.arange(r_test_range[0], r_test_range[1] + resolution, resolution)
    mi_iters = np.zeros(len(r_iters))
    neuropil_trace[np.isnan(neuropil_trace)] = 0
    raw_trace[np.isnan(raw_trace)] = 0
    for r_i, r_temp in enumerate(r_iters):
        Fc = raw_trace - r_temp * neuropil_trace
        mi_iters[r_i] = skimage.metrics.normalized_mutual_information(Fc, neuropil_trace)
    min_ind = np.argmin(mi_iters)
    r_best = r_iters[min_ind]
    return r_best, mi_iters, r_iters


def get_FC_from_r(raw_trace, neuropil_trace, min_r_count=5):
    """
    Get the corrected trace from the raw trace and neuropil trace using the given r values.

    Parameters
    ----------
    raw_trace : np.ndarray
        1D array of raw trace values.
    neuropil_trace : np.ndarray
        1D array of neuropil trace values.
    min_r_count : int
        Minimum number of r values to use for mean r value calculation.

    Returns
    -------
    FCs : np.ndarray
        1D array of corrected traces for each r value.
    r_values : np.ndarray
        1D array of r values used for the correction.
    raw_r : np.ndarray
        1D array of r values that minimized the mutual information.
    """
    r_values = np.zeros(raw_trace.shape[0])
    FCs = np.zeros_like(raw_trace)
    for roi in range(raw_trace.shape[0]):
        r_values[roi], _, _ = get_r_from_min_mi(raw_trace[roi], neuropil_trace[roi])
    mean_r = np.mean(r_values[r_values < 1])
    if len(np.where(r_values < 1)[0]) < min_r_count:
        mean_r = 0.8
    raw_r = r_values.copy()
    r_values[r_values >= 1] = mean_r
    for roi in range(raw_trace.shape[0]):
        FCs[roi] = raw_trace[roi] - r_values[roi] * neuropil_trace[roi]
    return FCs, r_values, raw_r


def make_output_directory(output_dir: Path, experiment_id: str) -> str:
    """Creates the output directory if it does not exist

    Parameters
    ----------
    output_dir: Path
        output directory
    experiment_id: str
        experiment_id number

    Returns
    -------
    output_dir: str
        output directory
    """
    output_dir = output_dir / experiment_id
    output_dir.mkdir(exist_ok=True)
    output_dir = output_dir / "extraction"
    output_dir.mkdir(exist_ok=True)
    return output_dir


def write_data_process(
    metadata: dict,
    input_fp: Union[str, Path],
    output_fp: Union[str, Path],
    start_time: dt,
    end_time: dt,
) -> None:
    """Writes output metadata to plane processing.json

    Parameters
    ----------
    metadata: dict
        parameters from suite2p motion correction
    raw_movie: str
        path to raw movies
    output_fp: str
        path to motion corrected movies
    """
    if isinstance(input_fp, Path):
        input_fp = str(input_fp)
    if isinstance(output_fp, Path):
        output_fp = str(output_fp)
    data_proc = DataProcess(
        name=ProcessName.VIDEO_ROI_TIMESERIES_EXTRACTION,
        software_version=os.getenv("VERSION", ""),
        start_date_time=start_time.isoformat(),
        end_date_time=end_time.isoformat(),
        input_location=input_fp,
        output_location=output_fp,
        code_url=(os.getenv("REPO_URL", "")),
        parameters=metadata,
    )
    if isinstance(output_fp, str):
        output_dir = Path(output_fp).parent
    with open(output_dir / "data_process.json", "w") as f:
        json.dump(json.loads(data_proc.model_dump_json()), f, indent=4)


def create_virtual_dataset(
    h5_file: Path, frame_locations: list, frames_length: int, temp_dir: Path
) -> Path:
    """Creates a virtual dataset from a list of frame locations

    Parameters
    ----------
    h5_file: Path
        path to h5 file
    frame_locations: list
        list of frame locations
    frames_length: int
        sum of frame locations
    temp_dir: Path
        temporary directory for virtual dataset

    Returns
    -------
    h5_file: Path
        path to virtual dataset
    """
    with h5py.File(h5_file, "r") as f:
        data_shape = f["data"].shape
        dtype = f["data"].dtype
        vsource = h5py.VirtualSource(f["data"])
        layout = h5py.VirtualLayout(shape=(frames_length, *data_shape[1:]), dtype=dtype)
        start = 0
        for loc in frame_locations:
            layout[start : start + loc[1] - loc[0] + 1] = vsource[loc[0] : loc[1] + 1]
            start += loc[1] - loc[0] + 1
        h5_file = temp_dir / h5_file.name
        with h5py.File(h5_file, "w") as f:
            f.create_virtual_dataset("data", layout)

    return h5_file


def bergamo_segmentation(motion_corr_fp: Path, session: dict, temp_dir: Path) -> str:
    """Performs singleplane motion correction on a singleplane data set

    Parameters
    ----------
    motion_corr_fp: Path
        path to data directory
    session: dict
        session information
    temp_dir: Path
        temporary directory for virtual dataset
    Returns
    -------
    h5_file: str
        path to motion corrected h5 file
    """
    motion_dir = motion_corr_fp.parent
    epoch_loc_fp = next(motion_dir.glob("epoch_locations.json"))
    with open(epoch_loc_fp, "r") as j:
        epoch_locations = json.load(j)
    valid_epoch_stems = [
        i["output_parameters"]["tiff_stem"]
        for i in session["stimulus_epochs"]
        if i["stimulus_name"] != "2p photostimulation"
    ]
    frame_locations = [epoch_locations[i] for i in valid_epoch_stems]
    frames_length = sum([(i[1] - i[0] + 1) for i in frame_locations])

    return create_virtual_dataset(
        motion_corr_fp, frame_locations, frames_length, temp_dir
    )


def get_metdata(input_dir: Path) -> Tuple[dict, dict, dict]:
    """Get the session and data description metadata from the input directory

    Parameters
    ----------
    input_dir: Path
        input directory

    Returns
    -------
    session: dict
        session metadata
    data_description: dict
        data description metadata
    processing: dict
        processing metadata
    subject: dict
        subject metadata
    """
    session_fp = next(input_dir.rglob("session.json"))
    with open(session_fp, "r") as j:
        session = json.load(j)
    data_des_fp = next(input_dir.rglob("data_description.json"))
    with open(data_des_fp, "r") as j:
        data_description = json.load(j)
    process_fp = next(input_dir.rglob("*/processing.json"))
    with open(process_fp, "r") as j:
        processing = json.load(j)
    session_fp = next(input_dir.rglob("subject.json"))
    with open(subject_fp, "r") as j:
        subject = json.load(j)

    return session, data_description, processing, subject


def get_frame_rate(processing: dict) -> float:
    """Get the frame rate from the processing metadata

    Parameters
    ----------
    processing: dict
        processing metadata

    Returns
    -------
    frame_rate: float
        frame rate
    """
    if processing.get("processing_pipeline") is not None:
        processing = processing["processing_pipeline"]
    frame_rate = None
    for data_proc in processing["data_processes"]:
        if data_proc["parameters"].get("movie_frame_rate_hz", ""):
            frame_rate = data_proc["parameters"]["movie_frame_rate_hz"]
    if frame_rate == None:
        raise ValueError("Frame rate not found in processing metadata")
    return frame_rate


def com(rois):
    """Calculation of the center of mass for spatial components

    Parameters
    ----------
    rois : np.ndarray or sparse.COO tensor
        Tensor of Spatial components (K x height x width)

    Returns
    -------
    cm : np.ndarray
        center of mass for spatial components (K x 2)
    """
    d1, d2 = rois.shape[1:]
    Coor = np.array(
        list(map(np.ravel, np.meshgrid(np.arange(d2), np.arange(d1)))), dtype=rois.dtype
    )
    A = rois.reshape((rois.shape[0], d1 * d2)).tocsc()
    return (A / A.sum(axis=1)).dot(Coor.T)


def get_contours(rois, thr=0.2, thr_method="max"):
    """Gets contour of spatial components and returns their coordinates

    Parameters
    ----------
    rois : np.ndarray or sparse.COO tensor
        Tensor of Spatial components (K x height x width)
    thr : float between 0 and 1, optional
        threshold for computing contours, by default 0.2
    thr_method : str, optional
        Method of thresholding:
            'max' sets to zero pixels that have value less than a fraction of the max value
            'nrg' keeps the pixels that contribute up to a specified fraction of the energy

    Returns
    -------
    coordinates : list
        list of coordinates with center of mass and contour plot coordinates for each component
    """

    nr, dims = rois.shape[0], rois.shape[1:]
    d1, d2 = dims[:2]
    d = np.prod(dims)
    x, y = np.mgrid[0:d1:1, 0:d2:1]

    coordinates = []

    # get the center of mass of neurons( patches )
    cm = com(rois)
    A = rois.T.reshape((d, nr)).tocsc()

    for i in range(nr):
        pars = dict()
        # we compute the cumulative sum of the energy of the Ath component that has been ordered from least to highest
        patch_data = A.data[A.indptr[i] : A.indptr[i + 1]]
        indx = np.argsort(patch_data)[::-1]
        if thr_method == "nrg":
            cumEn = np.cumsum(patch_data[indx] ** 2)
            if len(cumEn) == 0:
                pars = dict(
                    coordinates=np.array([]),
                    CoM=np.array([np.NaN, np.NaN]),
                    neuron_id=i + 1,
                )
                coordinates.append(pars)
                continue
            else:
                # we work with normalized values
                cumEn /= cumEn[-1]
                Bvec = np.ones(d)
                # we put it in a similar matrix
                Bvec[A.indices[A.indptr[i] : A.indptr[i + 1]][indx]] = cumEn
        else:
            if thr_method != "max":
                warn("Unknown threshold method. Choosing max")
            Bvec = np.zeros(d)
            Bvec[A.indices[A.indptr[i] : A.indptr[i + 1]]] = patch_data / patch_data.max()

        Bmat = np.reshape(Bvec, dims, order="F")
        pars["coordinates"] = []
        # for each dimensions we draw the contour
        for B in Bmat if len(dims) == 3 else [Bmat]:
            vertices = skimage.measure.find_contours(B.T, thr)
            # this fix is necessary for having disjoint figures and borders plotted correctly
            v = np.atleast_2d([np.nan, np.nan])
            for _, vtx in enumerate(vertices):
                num_close_coords = np.sum(np.isclose(vtx[0, :], vtx[-1, :]))
                if num_close_coords < 2:
                    if num_close_coords == 0:
                        # case angle
                        newpt = np.round(vtx[-1, :] / [d2, d1]) * [d2, d1]
                        vtx = np.concatenate((vtx, newpt[np.newaxis, :]), axis=0)
                    else:
                        # case one is border
                        vtx = np.concatenate((vtx, vtx[0, np.newaxis]), axis=0)
                v = np.concatenate((v, vtx, np.atleast_2d([np.nan, np.nan])), axis=0)

            pars["coordinates"] = v if len(dims) == 2 else (pars["coordinates"] + [v])
        pars["CoM"] = np.squeeze(cm[i, :])
        pars["neuron_id"] = i + 1
        coordinates.append(pars)
    return coordinates


def contour_video(
    output_path: str,
    data: Union[h5py.Dataset, np.ndarray],
    rois: Union[sparse.COO, np.ndarray],
    traces: np.ndarray,
    downscale: int = 10,
    fs: float = 30,
    lower_quantile: float = 0.02,
    upper_quantile: float = 0.9975,
    only_raw: bool = False,
    n_jobs: int = None if (tmp := os.environ.get("CO_CPUS")) is None else int(tmp),
    bitrate: str = "0",
    crf: int = 20,
    cpu_used: int = 4,
):
    """Create a video contours using vp9 codec via imageio-ffmpeg

    Parameters
    ----------
    output_path : str
        Desired output path for encoded video
    data : h5py.Dataset or numpy.ndarray
        Video to be encoded
    rois : np.ndarray or sparse.COO tensor
        Tensor of spatial components (K x height x width)
    traces: np.ndarray
        Tensor of temporal components (K x T)
    downscale : int = 10
        Decimation factor
    fs : float
        Desired frame rate for encoded video
    lower_quantile : float
        Lower cutoff value supplied to `np.quantile()` for normalization
    upper_quantile : float
        Upper cutoff value supplied to `np.quantile()` for normalization
    only_raw : bool, optional
        Produce video of raw data only, i.e. no reconstruction and residual
    n_jobs : int, optional
        The number of jobs to run in parallel.
    bitrate : str, optional
        Desired bitrate of output, by default "0". The default *MUST*
        be zero in order to encode in constant quality mode. Other values
        will result in constrained quality mode.
    crf : int, optional
        Desired perceptual quality of output, by default 20. Value can
        be from 0 - 63. Lower values mean better quality (but bigger video
        sizes).
    cpu_used : int, optional
        Sets how efficient the compression will be, by default 4. Values can
        be between 0 and 5. Higher values increase encoding speed at the
        expense of having some impact on quality and rate control accuracy.
    """
    dims = data.shape[1:]
    # create image of countours
    img_contours = np.zeros(dims + (3,), np.uint8)
    rgb = (255, 127, 14)
    for m in rois:
        if isinstance(m, sparse.COO):
            m = m.todense()
        ret, thresh = cv2.threshold(
            (m > m.max() / 10).astype(np.uint8), 0, 1, cv2.THRESH_BINARY
        )
        contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2]
        for contour in contours:
            cv2.drawContours(img_contours, contour, -1, rgb, max(max(dims) // 200, 1))
    # assemble movie tiles
    mov = downsample_array(data, downscale, 1, n_jobs=n_jobs)
    minmov, maxmov = np.nanquantile(
        mov[:: max(1, len(mov) // 100)], (lower_quantile, upper_quantile)
    )
    scale = lambda m: np.array(
        ThreadPool(n_jobs).map(
            lambda frame: np.clip(
                255 * (frame - minmov) / (maxmov - minmov), 0, 255
            ).astype(np.uint8),
            m,
        )
    )
    if only_raw:
        mov = scale(mov)
    else:
        img_contours = np.repeat(img_contours[..., None], 3, 0).reshape(
            dims[0], 3 * dims[1], -1
        )
        reconstructed = np.tensordot(
            downsample_array(traces.T, downscale, 1, n_jobs=n_jobs).astype("f4"),
            rois,
            1,
        )
        residual = scale(mov - reconstructed)
        mov = scale(mov)
        reconstructed = scale(reconstructed)
        mov = np.concatenate([mov, reconstructed, residual], 2)
        del reconstructed
        del residual
    # create canvas with labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    magnify = max(500 // dims[0], 1)
    h, w = dims[0] * magnify, dims[1] * magnify
    textheight = cv2.getTextSize("True", font, h / 600, max(h // 200, 1))[0][1]
    canvas_size = (
        int(np.ceil(h * 1.08 / 16)) * 16,
        # int(np.ceil(((w if only_raw else 3 * w) + 1.3 * textheight)  / 16)) * 16,
        int(np.ceil((w if only_raw else 3 * w) / 16)) * 16,
    )
    # textpad = canvas_size[1] - 3 * w  # left padding for vertical text
    pad = (canvas_size[1] - 3 * w) // 2
    canvas = np.zeros(canvas_size + (3,), np.uint8)
    for i in range(1 if only_raw else 3):
        # text = ("Original", "Reconstructed", "Residual")[i]
        text = ("Original", "ROI Activity", "Remainder")[i]
        fontscale = min(h / 600, w / 190)
        textsize = cv2.getTextSize(text, font, fontscale, max(h // 200, 1))[0]
        cv2.putText(
            canvas,
            text,
            (int(w * (0.49, 1.5, 2.51)[i] + pad - textsize[0] / 2), h // 25),
            font,
            fontscale,
            (255, 255, 255),
            max(h // 200, 1),
            cv2.LINE_4,
        )
    # create writer object
    writer = imageio_ffmpeg.write_frames(
        output_path,
        canvas_size[::-1],  # ffmpeg expects video shape in terms of: (width, height)
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        codec="libvpx-vp9",
        fps=fs,
        bitrate=bitrate,
        output_params=[
            "-crf",
            str(crf),
            "-row-mt",
            "1",
            "-cpu-used",
            str(cpu_used),
        ],
    )
    writer.send(None)  # Seed ffmpeg-imageio writer generator
    # overlay image of contours and write each frame
    if magnify > 1:
        img_contours = cv2.resize(img_contours, (0, 0), fx=magnify, fy=magnify)
    is_contours = img_contours != 0
    for frame in mov:
        if magnify > 1:
            frame = cv2.resize(frame, (0, 0), fx=magnify, fy=magnify)
        frame = np.repeat(frame[..., None], 3, 2)
        frame[is_contours] = img_contours[is_contours]
        canvas[-h:, -(w if only_raw else 3 * w) :] = frame
        writer.send(canvas)
    writer.close()


if __name__ == "__main__":
    start_time = dt.now()
    # Set the log level and name the logger
    logger = logging.getLogger("Source extraction using Suite2p with or without Cellpose")
    logger.setLevel(logging.INFO)

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input-dir", type=str, help="Input directory", default="../data/"
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, help="Output directory", default="../results/"
    )
    parser.add_argument(
        "--tmp_dir",
        type=str,
        default="/scratch",
        help="Directory into which to write temporary files "
        "produced by Suite2P (default: /scratch)",
    )
    parser.add_argument(
        "--diameter",
        type=int,
        default=0,
        help="Diameter that will be used for cellpose. "
        "If set to zero, diameter is estimated.",
    )
    parser.add_argument(
        "--anatomical_only",
        type=int,
        default=2,
        help="If greater than 0, specifies what to use Cellpose on. "
        "1: Will find masks on max projection image divided by mean image "
        "2: Will find masks on mean image "
        "3: Will find masks on enhanced mean image "
        "4: Will find masks on maximum projection image",
    )
    parser.add_argument(
        "--denoise",
        action="store_true",
        help="Whether or not binned movie should be denoised before cell detection.",
    )
    parser.add_argument(
        "--cellprob_threshold",
        type=float,
        default=0.0,
        help="Threshold for cell detection that will be used by cellpose.",
    )
    parser.add_argument(
        "--flow_threshold",
        type=float,
        default=1.5,
        help="Flow threshold that will be used by cellpose.",
    )
    parser.add_argument(
        "--spatial_hp_cp",
        type=int,
        default=0,
        help="Window for spatial high-pass filtering of image "
        "to be used for cellpose.",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="cyto",
        help="Path to pretrained model or string for model type "
        "(can be user’s model).",
    )
    parser.add_argument(
        "--use_suite2p_neuropil",
        action="store_true",
        help="Whether to use the fix weight provided by suite2p for neuropil \
        correction. If not, we use a mutual information based method.",
    )
    parser.add_argument(
        "--contour_video",
        action="store_true",
        help="Create a video overlaying the raw data, ROI activity, "
        "and remainder with ROI contours.",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    input_dir = Path(args.input_dir).resolve()
    tmp_dir = Path(args.tmp_dir).resolve()
    session, data_description, processing, subject = get_metdata(input_dir)
    subject_id = subject.get("subject_id", "")
    name = data_description.get("name", "")
    setup_logging(
        "aind-ophys-extraction-suite2p", mouse_id=subject_id, session_name=name
    )
    if next(input_dir.rglob("*decrosstalk.h5"), ""):
        input_fn = next(input_dir.rglob("*decrosstalk.h5"))
    else:
        input_fn = next(input_dir.rglob("*registered.h5"))
    parent_directory = input_fn.parent
    if session is not None and "Bergamo" in session["rig_id"]:
        motion_corrected_fn = bergamo_segmentation(input_fn, session, temp_dir=tmp_dir)
    else:
        motion_corrected_fn = input_fn
    if not data_description or "multiplane" in data_description.get("name", ""):
        unique_id = motion_corrected_fn.parent.parent.name
    else:
        unique_id = "_".join(str(data_description["name"]).split("_")[-3:])

    frame_rate = get_frame_rate(processing)

    output_dir = make_output_directory(output_dir, unique_id)
    # Set suite2p args.
    suite2p_args = suite2p.default_ops()
    # Overwrite the parameters for suite2p that are exposed
    suite2p_args["diameter"] = args.diameter
    suite2p_args["anatomical_only"] = args.anatomical_only
    suite2p_args["cellprob_threshold"] = args.cellprob_threshold
    suite2p_args["flow_threshold"] = args.flow_threshold
    suite2p_args["spatial_hp_cp"] = args.spatial_hp_cp
    suite2p_args["pretrained_model"] = args.pretrained_model
    suite2p_args["denoise"] = args.denoise
    suite2p_args["save_path0"] = args.tmp_dir
    # Here we overwrite the parameters for suite2p that will not change in our
    # processing pipeline. These are parameters that are not exposed to
    # minimize code length. Those are not set to default.
    suite2p_args["h5py"] = str(motion_corrected_fn)
    suite2p_args["data_path"] = []
    suite2p_args["roidetect"] = True
    suite2p_args["do_registration"] = 0
    suite2p_args["spikedetect"] = False
    suite2p_args["fs"] = frame_rate
    suite2p_args["neuropil_extract"] = True

    # determine nbinned from bin_duration and fs
    suite2p_args["bin_duration"] = 3.7  # The duration of time (in seconds) that
    # should be considered 1 bin for Suite2P ROI detection purposes. Requires
    # a valid value for 'fs' in order to derive an
    # nbinned Suite2P value. This allows consistent temporal downsampling
    # across movies with different lengths and/or frame rates.
    with h5py.File(suite2p_args["h5py"], "r") as f:
        nframes = f["data"].shape[0]
    bin_size = suite2p_args["bin_duration"] * suite2p_args["fs"]
    suite2p_args["nbinned"] = int(nframes / bin_size)
    logger.info(
        f"Movie has {nframes} frames collected at "
        f"{suite2p_args['fs']} Hz. "
        "To get a bin duration of "
        f"{suite2p_args['bin_duration']} "
        f"seconds, setting nbinned to "
        f"{suite2p_args['nbinned']}."
    )

    logger.info(f"running Suite2P v{suite2p.version}")
    try:
        suite2p.run_s2p(suite2p_args)
    except IndexError:  # raised when no ROIs found
        pass

    # load in the rois from the stat file and movie path for shape
    with h5py.File(str(motion_corrected_fn), "r") as open_vid:
        dims = open_vid["data"][0].shape
    if len(list(Path(args.tmp_dir).rglob("stat.npy"))):
        suite2p_stat_path = str(next(Path(args.tmp_dir).rglob("stat.npy")))
        suite2p_stats = np.load(suite2p_stat_path, allow_pickle=True)
        if session is not None and "Bergamo" in session["rig_id"]:
            # extract signals for all frames, not just those used for cell detection
            stat, traces_roi, traces_neuropil, _, _ = (
                suite2p.extraction.extraction_wrapper(
                    suite2p_stats, h5py.File(input_fn)["data"], ops=suite2p_args
                )
            )
        else:  # all frames have already been used for detection as well as extraction
            suite2p_f_path = str(next(Path(args.tmp_dir).rglob("F.npy")))
            suite2p_fneu_path = str(next(Path(args.tmp_dir).rglob("Fneu.npy")))
            traces_roi = np.load(suite2p_f_path, allow_pickle=True)
            traces_neuropil = np.load(suite2p_fneu_path, allow_pickle=True)
        iscell = np.load(str(next(Path(args.tmp_dir).rglob("iscell.npy"))))
        if args.use_suite2p_neuropil:
            traces_corrected = traces_roi - suite2p_args["neucoeff"] * traces_neuropil
            r_values = suite2p_args["neucoeff"] * np.ones(traces_roi.shape[0])
        else:
            traces_corrected, r_values, raw_r = get_FC_from_r(traces_roi, traces_neuropil)
        # convert ROIs to sparse COO 3D-tensor a la https://sparse.pydata.org/en/stable/construct.html
        data = []
        coords = []
        neuropil_coords = []
        for i, roi in enumerate(suite2p_stats):
            data.append(roi["lam"])
            coords.append(
                np.array(
                    [i * np.ones(len(roi["lam"])), roi["ypix"], roi["xpix"]],
                    dtype=np.int16,
                )
            )
            neuropil_coords.append(
                np.array(
                    [
                        i * np.ones(len(roi["neuropil_mask"])),
                        roi["neuropil_mask"] // dims[1],
                        roi["neuropil_mask"] % dims[1],
                    ],
                    dtype=np.int16,
                )
            )
        keys = list(suite2p_stats[0].keys())
        for k in ("ypix", "xpix", "lam", "neuropil_mask"):
            keys.remove(k)
        stat = {}
        for k in keys:
            stat[k] = [s[k] for s in suite2p_stats]
        data = np.concatenate(data)
        coords = np.hstack(coords)
        neuropil_coords = np.hstack(neuropil_coords)
        stat["soma_crop"] = np.concatenate(stat["soma_crop"])
        stat["overlap"] = np.concatenate(stat["overlap"])
    else:  # no ROIs found
        traces_roi, traces_neuropil, traces_corrected = [
            np.empty((0, nframes), dtype=np.float32)
        ] * 3
        r_values, data, coords, neuropil_coords, iscell = [[]] * 5
        if not args.use_suite2p_neuropil:
            raw_r = []
        keys = []

    cellpose_path = str(next(Path(args.tmp_dir).rglob("cellpose.npz"), ""))
    ops_path = str(next(Path(args.tmp_dir).rglob("ops.npy")))
    # write output files
    with h5py.File(output_dir / f"{unique_id}_extraction.h5", "w") as f:
        # traces
        f.create_dataset("traces/corrected", data=traces_corrected, compression="gzip")
        f.create_dataset("traces/neuropil", data=traces_neuropil, compression="gzip")
        f.create_dataset("traces/roi", data=traces_roi, compression="gzip")
        f.create_dataset("traces/neuropil_rcoef", data=r_values)
        if not args.use_suite2p_neuropil:
            # We save the raw r values if we are not using the suite2p neuropil.
            # This is useful for debugging purposes.
            f.create_dataset("traces/raw_neuropil_rcoef_mutualinfo", data=raw_r)
        for k in keys:
            dtype = np.array(stat[k]).dtype
            if dtype != "bool":
                dtype = "i2" if np.issubdtype(dtype, np.integer) else "f4"
            if k in ("skew", "std"):
                f.create_dataset(f"traces/{k}", data=stat[k], dtype=dtype)
            # ROIs
            else:
                f.create_dataset(f"rois/{k}", data=stat[k], dtype=dtype)
        f.create_dataset("rois/coords", data=coords, compression="gzip")
        f.create_dataset("rois/data", data=data, compression="gzip")
        shape = np.array([len(traces_roi), *dims], dtype=np.int16)
        f.create_dataset("rois/shape", data=shape)  # neurons x height x width
        f.create_dataset("rois/neuropil_coords", data=neuropil_coords, compression="gzip")
        # cellpose
        if cellpose_path:
            with np.load(cellpose_path) as cp:
                for k in cp.keys():
                    f.create_dataset(f"cellpose/{k}", data=cp[k], compression="gzip")
        else:
            logging.warning("No cellpose output found.")

        # classifier
        f.create_dataset(f"iscell", data=iscell, dtype="f4")
        # summary images
        ops = np.load(ops_path, allow_pickle=True)[()]
        f.create_dataset(f"meanImg", data=ops["meanImg"], compression="gzip")
        f.create_dataset(f"maxImg", data=ops["max_proj"], compression="gzip")

    write_data_process(
        vars(args),
        str(parent_directory),
        input_fn,
        output_dir / "extraction.h5",
        start_time,
        dt.now(),
    )

    # plot contours of detected ROIs over a selection of summary images
    rois = sparse.COO(coords, data, shape)
    cm = com(rois)
    coordinates = get_contours(rois)
    with h5py.File(str(motion_corrected_fn), "r") as f:
        corr_img = max_corr_image(f["data"])
    # plot
    x_size = 17 * max(dims[1] / dims[0], 0.4)
    fix, ax = plt.subplots(1, 3, figsize=(x_size, 6))
    lw = min(512 / max(*dims), 3)
    for i, img in enumerate((ops["meanImg"], ops["max_proj"], corr_img)):
        vmin, vmax = np.nanpercentile(img, (1, 99))
        ax[i].imshow(img, interpolation=None, cmap="gray", vmin=vmin, vmax=vmax)
        for c in coordinates:
            ax[i].plot(*c["coordinates"].T, c="orange", lw=lw)
        ax[i].axis("off")
        ax[i].set_title(
            ("mean image", "max image", "correlation image")[i],
            fontsize=min(24, 2.4 + 2 * x_size),
        )
    plt.tight_layout(pad=0.1)
    plt.savefig(
        output_dir / f"{unique_id}_detected_ROIs.png",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    for i in (0, 1, 2):
        for k in range(rois.shape[0]):
            ax[i].text(*cm[k], str(k), color="orange", fontsize=8 * lw)
    plt.savefig(
        output_dir / f"{unique_id}_detected_ROIs_withIDs.png",
        bbox_inches="tight",
        pad_inches=0.02,
    )

    if args.contour_video:
        with h5py.File(str(motion_corrected_fn), "r") as f:
            contour_video(
                output_dir / f"{unique_id}_ROI_contours_overlay.webm",
                f["data"],
                rois,
                traces_corrected,
                fs=frame_rate,
            )
