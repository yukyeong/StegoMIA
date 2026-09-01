"""Frequency-domain point-to-point image steganography utilities.

This module provides a lightweight, frequency-based embedding and
extraction routine that operates on the luminance channel of RGB
images. A shared integer key determines the block traversal order, so
communicating parties can recover the payload deterministically.

Example usage
------------
Embed a message into a cover image and save the stego result:
    python frequency_teganography.py encode cover.png stego.png \
        --text "secret" --key 1234

Extract a message from a stego image (using the same key):
    python frequency_teganography.py decode stego.png --key 1234
"""
from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
from PIL import Image
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    _TORCH_AVAILABLE = False


BlockCoords = Sequence[Tuple[int, int]]

_DCT_CACHE: Dict[int, np.ndarray] = {}


def _to_bits(message: bytes) -> np.ndarray:
    """Convert bytes to a bit array with a 32-bit length prefix."""
    length_prefix = np.array([len(message)], dtype=">u4").view(np.uint8)
    header_bits = np.unpackbits(length_prefix)
    payload_bits = np.unpackbits(np.frombuffer(message, dtype=np.uint8))
    return np.concatenate([header_bits, payload_bits]).astype(np.uint8)


def _from_bits(bits: np.ndarray) -> bytes:
    """Convert a bit array with a 32-bit length prefix back to bytes."""
    if bits.size < 32:
        raise ValueError("Insufficient bits to decode length header.")

    length = np.packbits(bits[:32]).view(">u4")[0]
    total_bits = 32 + int(length) * 8
    if bits.size < total_bits:
        raise ValueError("Bit stream shorter than advertised message length.")

    payload_bits = bits[32:total_bits]
    return np.packbits(payload_bits).tobytes()


def _pad_channel(channel: np.ndarray, block_size: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Pad channel so height and width are divisible by block_size."""
    h, w = channel.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    if pad_h == 0 and pad_w == 0:
        return channel, (0, 0)

    padded = np.pad(channel, ((0, pad_h), (0, pad_w)), mode="reflect")
    return padded, (pad_h, pad_w)


def _generate_block_coords(height: int, width: int, block_size: int) -> BlockCoords:
    rows = range(0, height, block_size)
    cols = range(0, width, block_size)
    return [(r, c) for r in rows for c in cols]


def _fft2_block(block: np.ndarray) -> np.ndarray:
    return np.fft.fft2(block)


def _ifft2_block(block: np.ndarray) -> np.ndarray:
    return np.fft.ifft2(block)


def _dct_matrix(size: int) -> np.ndarray:
    """Build an orthonormal DCT (type-II) matrix."""
    if size in _DCT_CACHE:
        return _DCT_CACHE[size]
    n = np.arange(size)
    k = n.reshape(-1, 1)
    mat = np.cos((np.pi / (2 * size)) * (2 * n + 1) * k).astype(np.float32)
    mat[0, :] *= np.sqrt(1.0 / size)
    mat[1:, :] *= np.sqrt(2.0 / size)
    _DCT_CACHE[size] = mat
    return mat


def _dct2_block(block: np.ndarray, mat: np.ndarray) -> np.ndarray:
    return mat @ block @ mat.T


def _idct2_block(block: np.ndarray, mat: np.ndarray) -> np.ndarray:
    # Inverse for orthonormal DCT-II is DCT-III.
    return mat.T @ block @ mat


def _resolve_dct_device(device: Optional[str]) -> Optional[str]:
    if not _TORCH_AVAILABLE:
        return None
    if device is None or device == "auto":
        return "cuda" if torch.cuda.is_available() else None
    device_lower = device.lower()
    if device_lower in ("cuda", "gpu") or device_lower.startswith("cuda"):
        return device if torch.cuda.is_available() else None
    return None


def _embed_frequency_dct_torch(
    cover: Image.Image,
    message: bytes,
    *,
    key: int = 0,
    block_size: int = 8,
    freq: Tuple[int, int] = (3, 3),
    min_magnitude: float = 2.0,
    center: bool = True,
    device: str = "cuda",
) -> Image.Image:
    """Embed a message into the luminance channel via blockwise DCT on GPU."""
    if cover.mode != "RGB":
        cover = cover.convert("RGB")

    y, cb, cr = cover.convert("YCbCr").split()
    y_arr = np.asarray(y, dtype=np.float32)
    padded, (pad_h, pad_w) = _pad_channel(y_arr, block_size)
    if center:
        padded = padded - 128.0

    bits = _to_bits(message)
    height, width = padded.shape
    num_blocks = (height // block_size) * (width // block_size)
    if bits.size > num_blocks:
        raise ValueError(
            f"Message too long: need {bits.size} blocks but only {num_blocks} available.")

    rng = np.random.default_rng(key)
    permutation = rng.permutation(num_blocks)
    freq_r, freq_c = freq

    with torch.no_grad():
        device_obj = torch.device(device)
        y_tensor = torch.from_numpy(padded).to(device=device_obj, dtype=torch.float32)
        y_tensor = y_tensor.unsqueeze(0).unsqueeze(0)

        unfold = torch.nn.Unfold(kernel_size=block_size, stride=block_size)
        blocks = unfold(y_tensor)
        blocks = blocks.transpose(1, 2).reshape(num_blocks, block_size, block_size)

        dct_mat = torch.tensor(_dct_matrix(block_size), device=device_obj, dtype=torch.float32)
        coeffs = torch.matmul(dct_mat, blocks)
        coeffs = torch.matmul(coeffs, dct_mat.t())

        perm = torch.as_tensor(permutation[:bits.size], device=device_obj, dtype=torch.long)
        bits_t = torch.as_tensor(bits, device=device_obj, dtype=torch.float32)
        magnitude = coeffs[perm, freq_r, freq_c].abs().clamp(min=min_magnitude)
        sign = torch.where(bits_t > 0, 1.0, -1.0)
        coeffs[perm, freq_r, freq_c] = sign * magnitude

        blocks_mod = torch.matmul(dct_mat.t(), coeffs)
        blocks_mod = torch.matmul(blocks_mod, dct_mat)
        if center:
            blocks_mod = blocks_mod + 128.0

        blocks_mod = blocks_mod.reshape(1, num_blocks, block_size * block_size).transpose(1, 2)
        fold = torch.nn.Fold(output_size=(height, width), kernel_size=block_size, stride=block_size)
        y_mod = fold(blocks_mod).squeeze(0).squeeze(0)
        y_mod = y_mod.clamp(0, 255).to(torch.uint8).cpu().numpy()

    if pad_h or pad_w:
        y_mod = y_mod[:height - pad_h, :width - pad_w]

    stego_ycbcr = Image.merge("YCbCr", (Image.fromarray(y_mod), cb, cr))
    return stego_ycbcr.convert("RGB")


def embed_frequency(
    cover: Image.Image,
    message: bytes,
    *,
    key: int = 0,
    block_size: int = 8,
    freq: Tuple[int, int] = (3, 3),
    min_magnitude: float = 2.0,
) -> Image.Image:
    """Embed a message into the luminance channel via blockwise FFT.

    Args:
        cover: RGB cover image.
        message: Bytes payload to embed.
        key: Shared integer key used to permute block traversal.
        block_size: Spatial block size for frequency manipulation.
        freq: Target frequency coordinate within each block (avoid DC).
        min_magnitude: Minimum magnitude enforced on modified frequency.
    """
    if cover.mode != "RGB":
        cover = cover.convert("RGB")

    y, cb, cr = cover.convert("YCbCr").split()
    y_arr = np.asarray(y, dtype=np.float32)
    padded, (pad_h, pad_w) = _pad_channel(y_arr, block_size)

    bits = _to_bits(message)
    block_coords = _generate_block_coords(*padded.shape, block_size=block_size)
    if bits.size > len(block_coords):
        raise ValueError(
            f"Message too long: need {bits.size} blocks but only {len(block_coords)} available.")

    rng = np.random.default_rng(key)
    permutation = rng.permutation(len(block_coords))

    freq_r, freq_c = freq
    modified = padded.copy()
    for bit, idx in zip(bits, permutation):
        r, c = block_coords[idx]
        block = modified[r:r + block_size, c:c + block_size]
        spectrum = _fft2_block(block)
        coef = spectrum[freq_r, freq_c]
        magnitude = max(abs(coef), min_magnitude)
        phase = np.angle(coef)
        sign = 1.0 if bit else -1.0
        spectrum[freq_r, freq_c] = sign * magnitude * np.exp(1j * phase)
        block_out = _ifft2_block(spectrum).real
        modified[r:r + block_size, c:c + block_size] = block_out

    if pad_h or pad_w:
        modified = modified[:modified.shape[0] - pad_h, :modified.shape[1] - pad_w]

    modified = np.clip(modified, 0, 255).astype(np.uint8)
    stego_ycbcr = Image.merge("YCbCr", (Image.fromarray(modified), cb, cr))
    return stego_ycbcr.convert("RGB")


def embed_frequency_dct(
    cover: Image.Image,
    message: bytes,
    *,
    key: int = 0,
    block_size: int = 8,
    freq: Tuple[int, int] = (3, 3),
    min_magnitude: float = 2.0,
    center: bool = True,
    device: Optional[str] = None,
) -> Image.Image:
    """Embed a message into the luminance channel via blockwise DCT.

    Uses real-valued DCT coefficients for a stable sign-based encoding.
    """
    torch_device = _resolve_dct_device(device)
    if torch_device is not None:
        return _embed_frequency_dct_torch(
            cover,
            message,
            key=key,
            block_size=block_size,
            freq=freq,
            min_magnitude=min_magnitude,
            center=center,
            device=torch_device,
        )
    if cover.mode != "RGB":
        cover = cover.convert("RGB")

    y, cb, cr = cover.convert("YCbCr").split()
    y_arr = np.asarray(y, dtype=np.float32)
    padded, (pad_h, pad_w) = _pad_channel(y_arr, block_size)
    if center:
        padded = padded - 128.0

    bits = _to_bits(message)
    block_coords = _generate_block_coords(*padded.shape, block_size=block_size)
    if bits.size > len(block_coords):
        raise ValueError(
            f"Message too long: need {bits.size} blocks but only {len(block_coords)} available.")

    rng = np.random.default_rng(key)
    permutation = rng.permutation(len(block_coords))

    freq_r, freq_c = freq
    dct_mat = _dct_matrix(block_size)
    modified = padded.copy()
    for bit, idx in zip(bits, permutation):
        r, c = block_coords[idx]
        block = modified[r:r + block_size, c:c + block_size]
        coeffs = _dct2_block(block, dct_mat)
        magnitude = max(abs(coeffs[freq_r, freq_c]), min_magnitude)
        coeffs[freq_r, freq_c] = (1.0 if bit else -1.0) * magnitude
        modified[r:r + block_size, c:c + block_size] = _idct2_block(coeffs, dct_mat)

    if center:
        modified = modified + 128.0

    if pad_h or pad_w:
        modified = modified[:modified.shape[0] - pad_h, :modified.shape[1] - pad_w]

    modified = np.clip(modified, 0, 255).astype(np.uint8)
    stego_ycbcr = Image.merge("YCbCr", (Image.fromarray(modified), cb, cr))
    return stego_ycbcr.convert("RGB")


def extract_frequency(
    stego: Image.Image,
    *,
    key: int = 0,
    block_size: int = 8,
    freq: Tuple[int, int] = (3, 3),
) -> bytes:
    """Extract a message from a stego image using the shared key."""
    if stego.mode != "RGB":
        stego = stego.convert("RGB")

    y = stego.convert("YCbCr").split()[0]
    y_arr = np.asarray(y, dtype=np.float32)
    padded, _ = _pad_channel(y_arr, block_size)

    block_coords = _generate_block_coords(*padded.shape, block_size=block_size)
    rng = np.random.default_rng(key)
    permutation = rng.permutation(len(block_coords))

    freq_r, freq_c = freq
    recovered_bits = []
    # Recover length header first
    for idx in permutation[:32]:
        r, c = block_coords[idx]
        spectrum = _fft2_block(padded[r:r + block_size, c:c + block_size])
        recovered_bits.append(1 if spectrum[freq_r, freq_c].real >= 0 else 0)

    length = np.packbits(np.array(recovered_bits, dtype=np.uint8)).view(">u4")[0]
    total_bits = 32 + int(length) * 8
    if total_bits > len(block_coords):
        raise ValueError("Encoded payload exceeds available blocks; wrong key or parameters?")

    for idx in permutation[32:total_bits]:
        r, c = block_coords[idx]
        spectrum = _fft2_block(padded[r:r + block_size, c:c + block_size])
        recovered_bits.append(1 if spectrum[freq_r, freq_c].real >= 0 else 0)

    return _from_bits(np.array(recovered_bits, dtype=np.uint8))


def extract_frequency_dct(
    stego: Image.Image,
    *,
    key: int = 0,
    block_size: int = 8,
    freq: Tuple[int, int] = (3, 3),
    center: bool = True,
) -> bytes:
    """Extract a message from a stego image using blockwise DCT."""
    if stego.mode != "RGB":
        stego = stego.convert("RGB")

    y = stego.convert("YCbCr").split()[0]
    y_arr = np.asarray(y, dtype=np.float32)
    padded, _ = _pad_channel(y_arr, block_size)
    if center:
        padded = padded - 128.0

    block_coords = _generate_block_coords(*padded.shape, block_size=block_size)
    rng = np.random.default_rng(key)
    permutation = rng.permutation(len(block_coords))

    freq_r, freq_c = freq
    dct_mat = _dct_matrix(block_size)
    recovered_bits = []

    for idx in permutation[:32]:
        r, c = block_coords[idx]
        coeffs = _dct2_block(padded[r:r + block_size, c:c + block_size], dct_mat)
        recovered_bits.append(1 if coeffs[freq_r, freq_c] >= 0 else 0)

    length = np.packbits(np.array(recovered_bits, dtype=np.uint8)).view(">u4")[0]
    total_bits = 32 + int(length) * 8
    if total_bits > len(block_coords):
        raise ValueError("Encoded payload exceeds available blocks; wrong key or parameters?")

    for idx in permutation[32:total_bits]:
        r, c = block_coords[idx]
        coeffs = _dct2_block(padded[r:r + block_size, c:c + block_size], dct_mat)
        recovered_bits.append(1 if coeffs[freq_r, freq_c] >= 0 else 0)

    return _from_bits(np.array(recovered_bits, dtype=np.uint8))


def _load_message(text: Optional[str], path: Optional[Path]) -> bytes:
    if text is not None:
        return text.encode("utf-8")
    if path is not None:
        return path.read_bytes()
    raise ValueError("Provide --text or --message-file to supply payload.")


def _encode_cli(args: argparse.Namespace) -> None:
    cover = Image.open(args.cover)
    payload = _load_message(args.text, args.message_file)
    stego = embed_frequency(
        cover,
        payload,
        key=args.key,
        block_size=args.block_size,
        freq=tuple(args.freq),
        min_magnitude=args.min_magnitude,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stego.save(args.output)
    print(f"Embedded {len(payload)} bytes into {args.output} using key {args.key}.")


def _decode_cli(args: argparse.Namespace) -> None:
    stego = Image.open(args.stego)
    payload = extract_frequency(
        stego,
        key=args.key,
        block_size=args.block_size,
        freq=tuple(args.freq),
    )
    if args.output is None:
        print(payload.decode("utf-8", errors="replace"))
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(f"Recovered {len(payload)} bytes to {args.output} using key {args.key}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "Point-to-point frequency-domain steganography",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    enc = subparsers.add_parser("encode", help="Embed a payload into a cover image.")
    enc.add_argument("cover", type=Path, help="Path to the cover image.")
    enc.add_argument("output", type=Path, help="Path to save the stego image.")
    enc.add_argument("--text", type=str, help="Text payload to embed.")
    enc.add_argument("--message-file", type=Path, help="Binary payload file to embed.")
    enc.add_argument("--key", type=int, default=0, help="Shared integer key for block permutation.")
    enc.add_argument("--block-size", type=int, default=8, help="Block size for FFT embedding.")
    enc.add_argument(
        "--freq",
        nargs=2,
        type=int,
        default=(3, 3),
        metavar=("ROW", "COL"),
        help="Frequency coordinate inside each block used for embedding.",
    )
    enc.add_argument(
        "--min-magnitude",
        type=float,
        default=2.0,
        help="Minimum magnitude enforced on modified frequency coefficients.",
    )
    enc.set_defaults(func=_encode_cli)

    dec = subparsers.add_parser("decode", help="Extract a payload from a stego image.")
    dec.add_argument("stego", type=Path, help="Path to the stego image.")
    dec.add_argument("--key", type=int, default=0, help="Shared integer key for block permutation.")
    dec.add_argument("--block-size", type=int, default=8, help="Block size for FFT embedding.")
    dec.add_argument(
        "--freq",
        nargs=2,
        type=int,
        default=(3, 3),
        metavar=("ROW", "COL"),
        help="Frequency coordinate inside each block used for embedding.",
    )
    dec.add_argument("--output", type=Path, help="Optional path to save the recovered payload.")
    dec.set_defaults(func=_decode_cli)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
