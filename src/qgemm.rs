//! Fused multi-row quantized matrix multiplication.
//!
//! A multi-row pass multiplies one compressed weight matrix by several input
//! rows. Calling a per-row dot product once per input row keeps the weight
//! bytes in cache, but it re-runs the *block decode* — nibble unpacking and
//! scale extraction — once per input row, and that decode is roughly half the
//! per-row work. These kernels decode each weight block once and apply it to
//! every input row while the decoded form is still in registers.
//!
//! Candle keeps its block fields `pub(crate)`, so the layout mirrors below
//! re-declare the same `repr(C)` structures. Those layouts are GGUF's on-disk
//! block formats rather than candle internals, and every one is pinned by a
//! compile-time size assertion against candle's own type plus a differential
//! test against candle's `vec_dot` output.
//!
//! Accumulation order is chosen to match `vec_dot` exactly: the per-block
//! integer sums are associative, and the f32 accumulators are updated once per
//! block in the same sequence, so these kernels reproduce candle's results
//! bit for bit rather than merely closely.

use candle_core::quantized::k_quants::{BlockQ4K, BlockQ5K, BlockQ6K, BlockQ8_0, BlockQ8K};

/// Rows processed per register tile. Each tile holds one i32 accumulator and
/// one f32 accumulator per row alongside the decoded weight registers, so the
/// working set has to stay inside the 16 architectural ymm registers.
pub const TILE: usize = 8;

macro_rules! mirror {
    ($(#[$meta:meta])* $name:ident, $candle:ty, { $($field:ident : $ty:ty),* $(,)? }) => {
        $(#[$meta])*
        #[repr(C)]
        #[derive(Clone, Copy)]
        pub struct $name {
            $(pub $field: $ty),*
        }
        const _: () = assert!(
            std::mem::size_of::<$name>() == std::mem::size_of::<$candle>(),
            concat!(stringify!($name), " does not match the candle block layout"),
        );
        const _: () = assert!(
            std::mem::align_of::<$name>() <= std::mem::align_of::<$candle>(),
        );
    };
}

mirror!(
    /// `d * scale[sub] * q - dmin * min[sub]`, 256 weights in 8 sub-blocks.
    Q4KBlock, BlockQ4K, { d: u16, dmin: u16, scales: [u8; 12], qs: [u8; 128] }
);
mirror!(
    /// Q4K plus one high bit per weight in `qh`.
    Q5KBlock, BlockQ5K, { d: u16, dmin: u16, scales: [u8; 12], qh: [u8; 32], qs: [u8; 128] }
);
mirror!(
    /// Six bits per weight: four in `ql`, two in `qh`, with signed scales.
    Q6KBlock, BlockQ6K, { ql: [u8; 128], qh: [u8; 64], scales: [i8; 16], d: u16 }
);
mirror!(
    /// 32 signed weights and one scale.
    Q80Block, BlockQ8_0, { d: u16, qs: [i8; 32] }
);
mirror!(
    /// Quantized activations for the K-quant kernels, with per-16 sub-sums.
    Q8KBlock, BlockQ8K, { d: f32, qs: [i8; 256], bsums: [i16; 16] }
);

/// Reinterpret a slice of candle blocks as the matching layout mirror.
///
/// # Safety
///
/// `Mirror` must be the layout mirror of `Block`, which the `mirror!` macro's
/// size assertion enforces at compile time for every pair declared above.
pub(crate) unsafe fn as_mirror<Block, Mirror>(blocks: &[Block]) -> &[Mirror] {
    debug_assert_eq!(
        std::mem::size_of::<Block>(),
        std::mem::size_of::<Mirror>(),
        "layout mirror size mismatch"
    );
    // SAFETY: the two types have identical size and layout, the mirror is a
    // plain `repr(C)` POD for which every bit pattern is valid, and the
    // resulting slice borrows the same memory for the same lifetime.
    unsafe { std::slice::from_raw_parts(blocks.as_ptr().cast::<Mirror>(), blocks.len()) }
}

#[cfg(target_arch = "x86_64")]
pub fn supported() -> bool {
    use std::sync::OnceLock;
    static SUPPORTED: OnceLock<bool> = OnceLock::new();
    *SUPPORTED.get_or_init(|| {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    })
}

#[cfg(not(target_arch = "x86_64"))]
pub fn supported() -> bool {
    false
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::{Q4KBlock, Q6KBlock, Q8KBlock, Q80Block, TILE};
    use half::f16;
    use std::arch::x86_64::*;

    /// Broadcast the `index`-th of eight i16 scales across both 128-bit lanes.
    /// Byte-for-byte candle's `get_scale_shuffle_k4` table.
    #[rustfmt::skip]
    const SCALE_SHUFFLE: [u8; 256] = {
        let mut table = [0u8; 256];
        let mut index = 0;
        while index < 8 {
            let mut byte = 0;
            while byte < 32 {
                table[index * 32 + byte] = (index * 2) as u8;
                table[index * 32 + byte + 1] = (index * 2 + 1) as u8;
                byte += 2;
            }
            index += 1;
        }
        table
    };

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn scale_shuffle(index: usize) -> __m256i {
        // SAFETY: `index < 8` and the table holds eight 32-byte entries.
        unsafe { _mm256_loadu_si256((SCALE_SHUFFLE.as_ptr() as *const __m256i).add(index)) }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    unsafe fn hsum_f32_8(x: __m256) -> f32 {
        // Register-only reduction, identical to candle's hsum_float_8.
        let res = _mm256_extractf128_ps(x, 1);
        let res = _mm_add_ps(res, _mm256_castps256_ps128(x));
        let res = _mm_add_ps(res, _mm_movehl_ps(res, res));
        let res = _mm_add_ss(res, _mm_movehdup_ps(res));
        _mm_cvtss_f32(res)
    }

    #[inline]
    #[target_feature(enable = "avx")]
    unsafe fn hsum_f32_4(x: __m128) -> f32 {
        // Register-only reduction, matching candle's tail sequence.
        let res = _mm_add_ps(x, _mm_movehl_ps(x, x));
        let res = _mm_add_ss(res, _mm_movehdup_ps(res));
        _mm_cvtss_f32(res)
    }

    /// Unpack the twelve packed scale bytes into eight scales then eight mins,
    /// reproducing candle's `utmp` sequence exactly.
    #[inline]
    fn unpack_k_scales(scales: &[u8; 12]) -> [u32; 4] {
        const KMASK1: u32 = 0x3f3f3f3f;
        const KMASK2: u32 = 0x0f0f0f0f;
        const KMASK3: u32 = 0x03030303;
        let mut utmp = [0u32; 4];
        for (slot, chunk) in utmp[..3].iter_mut().zip(scales.chunks_exact(4)) {
            *slot = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
        let uaux = utmp[1] & KMASK1;
        utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
        utmp[2] = uaux;
        utmp[0] &= KMASK1;
        utmp
    }

    /// One tile of `T` input rows against a whole compressed weight row.
    ///
    /// The block decode — scale unpack, nibble split, scale broadcasts — runs
    /// once per block here and feeds all `T` rows, which is the entire point
    /// of the kernel.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn q4k_tile<const T: usize>(
        weight: &[Q4KBlock],
        inputs: &[Q8KBlock],
        rows: usize,
        base: usize,
        out: &mut [f32],
    ) {
        // SAFETY: callers pass `base + T <= rows`, `inputs` holds
        // `rows * blocks_per_row` blocks, and every load below is an unaligned
        // intrinsic inside those bounds.
        unsafe {
            let m4 = _mm256_set1_epi8(0xF);
            let mut acc = [_mm256_setzero_ps(); T];
            let mut acc_m = [_mm_setzero_ps(); T];

            for (block, w) in weight.iter().enumerate() {
                let utmp = unpack_k_scales(&w.scales);
                let mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(
                    utmp[3] as i32,
                    utmp[2] as i32,
                    utmp[1] as i32,
                    utmp[0] as i32,
                ));
                let sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
                let mins128 = _mm256_extracti128_si256(mins_and_scales, 1);
                let scales = _mm256_insertf128_si256(_mm256_castsi128_si256(sc128), sc128, 1);
                let xd = f16::from_bits(w.d).to_f32();
                let xdmin = f16::from_bits(w.dmin).to_f32();

                let mut sumi = [_mm256_setzero_si256(); T];
                for j in 0..4 {
                    let q4bits = _mm256_loadu_si256(w.qs.as_ptr().add(j * 32) as *const __m256i);
                    let q4l = _mm256_and_si256(q4bits, m4);
                    let q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
                    let scale_l = scale_shuffle(2 * j);
                    let scale_h = scale_shuffle(2 * j + 1);
                    let scale_l = _mm256_shuffle_epi8(scales, scale_l);
                    let scale_h = _mm256_shuffle_epi8(scales, scale_h);
                    for (lane, sumi) in sumi.iter_mut().enumerate() {
                        let y = inputs.get_unchecked(block * rows + base + lane);
                        let q8l = _mm256_loadu_si256(y.qs.as_ptr().add(j * 64) as *const __m256i);
                        let q8h =
                            _mm256_loadu_si256(y.qs.as_ptr().add(j * 64 + 32) as *const __m256i);
                        let p16l = _mm256_madd_epi16(scale_l, _mm256_maddubs_epi16(q4l, q8l));
                        let p16h = _mm256_madd_epi16(scale_h, _mm256_maddubs_epi16(q4h, q8h));
                        // Integer addition is associative, so folding the two
                        // halves before accumulating matches candle exactly.
                        *sumi = _mm256_add_epi32(*sumi, _mm256_add_epi32(p16l, p16h));
                    }
                }

                for (lane, (acc, acc_m)) in acc.iter_mut().zip(acc_m.iter_mut()).enumerate() {
                    let y = inputs.get_unchecked(block * rows + base + lane);
                    let d = y.d * xd;
                    let dmin = -y.d * xdmin;
                    *acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi[lane]), *acc);
                    let q8sums = _mm256_loadu_si256(y.bsums.as_ptr() as *const __m256i);
                    let q8s = _mm_hadd_epi16(
                        _mm256_extracti128_si256(q8sums, 0),
                        _mm256_extracti128_si256(q8sums, 1),
                    );
                    let prod = _mm_madd_epi16(mins128, q8s);
                    *acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), *acc_m);
                }
            }

            for (lane, (acc, acc_m)) in acc.iter().zip(&acc_m).enumerate() {
                *out.get_unchecked_mut(base + lane) = hsum_f32_8(*acc) + hsum_f32_4(*acc_m);
            }
        }
    }

    /// Q6K: six bits per weight, signed per-16 scales, no mins term.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn q6k_tile<const T: usize>(
        weight: &[Q6KBlock],
        inputs: &[Q8KBlock],
        rows: usize,
        base: usize,
        out: &mut [f32],
    ) {
        // SAFETY: as `q4k_tile`.
        unsafe {
            let m4 = _mm256_set1_epi8(0xF);
            let m2 = _mm256_set1_epi8(3);
            let m32s = _mm256_set1_epi8(32);
            let mut acc = [_mm256_setzero_ps(); T];

            for (block, w) in weight.iter().enumerate() {
                let xd = f16::from_bits(w.d).to_f32();
                let scales = _mm_loadu_si128(w.scales.as_ptr() as *const __m128i);
                let mut sumi = [_mm256_setzero_si256(); T];

                for j in 0..2 {
                    let q4bits1 = _mm256_loadu_si256(w.ql.as_ptr().add(j * 64) as *const __m256i);
                    let q4bits2 =
                        _mm256_loadu_si256(w.ql.as_ptr().add(j * 64 + 32) as *const __m256i);
                    let q4bits_h = _mm256_loadu_si256(w.qh.as_ptr().add(j * 32) as *const __m256i);

                    let q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bits_h, m2), 4);
                    let q4h_1 =
                        _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bits_h, 2), m2), 4);
                    let q4h_2 =
                        _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bits_h, 4), m2), 4);
                    let q4h_3 =
                        _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bits_h, 6), m2), 4);

                    let q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m4), q4h_0);
                    let q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m4), q4h_1);
                    let q4_2 =
                        _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m4), q4h_2);
                    let q4_3 =
                        _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m4), q4h_3);

                    let is = j * 4;
                    let scale_0 =
                        _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, q6k_scale_shuffle(is)));
                    let scale_1 =
                        _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, q6k_scale_shuffle(is + 1)));
                    let scale_2 =
                        _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, q6k_scale_shuffle(is + 2)));
                    let scale_3 =
                        _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, q6k_scale_shuffle(is + 3)));

                    for (lane, sumi) in sumi.iter_mut().enumerate() {
                        let y = inputs.get_unchecked(block * rows + base + lane);
                        let base_ptr = y.qs.as_ptr().add(j * 128);
                        let q8_0 = _mm256_loadu_si256(base_ptr as *const __m256i);
                        let q8_1 = _mm256_loadu_si256(base_ptr.add(32) as *const __m256i);
                        let q8_2 = _mm256_loadu_si256(base_ptr.add(64) as *const __m256i);
                        let q8_3 = _mm256_loadu_si256(base_ptr.add(96) as *const __m256i);

                        let q8s_0 = _mm256_maddubs_epi16(m32s, q8_0);
                        let q8s_1 = _mm256_maddubs_epi16(m32s, q8_1);
                        let q8s_2 = _mm256_maddubs_epi16(m32s, q8_2);
                        let q8s_3 = _mm256_maddubs_epi16(m32s, q8_3);

                        let p16_0 = _mm256_sub_epi16(_mm256_maddubs_epi16(q4_0, q8_0), q8s_0);
                        let p16_1 = _mm256_sub_epi16(_mm256_maddubs_epi16(q4_1, q8_1), q8s_1);
                        let p16_2 = _mm256_sub_epi16(_mm256_maddubs_epi16(q4_2, q8_2), q8s_2);
                        let p16_3 = _mm256_sub_epi16(_mm256_maddubs_epi16(q4_3, q8_3), q8s_3);

                        let p16_0 = _mm256_madd_epi16(scale_0, p16_0);
                        let p16_1 = _mm256_madd_epi16(scale_1, p16_1);
                        let p16_2 = _mm256_madd_epi16(scale_2, p16_2);
                        let p16_3 = _mm256_madd_epi16(scale_3, p16_3);

                        *sumi = _mm256_add_epi32(
                            *sumi,
                            _mm256_add_epi32(
                                _mm256_add_epi32(p16_0, p16_1),
                                _mm256_add_epi32(p16_2, p16_3),
                            ),
                        );
                    }
                }

                for (lane, acc) in acc.iter_mut().enumerate() {
                    let y = inputs.get_unchecked(block * rows + base + lane);
                    *acc = _mm256_fmadd_ps(
                        _mm256_set1_ps(y.d * xd),
                        _mm256_cvtepi32_ps(sumi[lane]),
                        *acc,
                    );
                }
            }

            for (lane, acc) in acc.iter().enumerate() {
                *out.get_unchecked_mut(base + lane) = hsum_f32_8(*acc);
            }
        }
    }

    /// Select scales `2 * index` and `2 * index + 1` into the low and high
    /// halves of an xmm register. Byte-for-byte candle's `get_scale_shuffle`.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn q6k_scale_shuffle(index: usize) -> __m128i {
        const TABLE: [u8; 128] = {
            let mut table = [0u8; 128];
            let mut byte = 0;
            while byte < 128 {
                table[byte] = (byte / 8) as u8;
                byte += 1;
            }
            table
        };
        // SAFETY: `index < 8` and the table holds eight 16-byte entries.
        unsafe { _mm_loadu_si128((TABLE.as_ptr() as *const __m128i).add(index)) }
    }

    /// Q8_0: activations and weights share the block format; only the absolute
    /// value of the weights and their sign application are hoistable.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn q80_tile<const T: usize>(
        weight: &[Q80Block],
        inputs: &[Q80Block],
        rows: usize,
        base: usize,
        out: &mut [f32],
    ) {
        // SAFETY: as `q4k_tile`.
        unsafe {
            let ones = _mm256_set1_epi16(1);
            let mut acc = [_mm256_setzero_ps(); T];
            for (block, w) in weight.iter().enumerate() {
                let bx = _mm256_loadu_si256(w.qs.as_ptr() as *const __m256i);
                let ax = _mm256_sign_epi8(bx, bx);
                let xd = f16::from_bits(w.d).to_f32();
                for (lane, acc) in acc.iter_mut().enumerate() {
                    let y = inputs.get_unchecked(block * rows + base + lane);
                    let by = _mm256_loadu_si256(y.qs.as_ptr() as *const __m256i);
                    let sy = _mm256_sign_epi8(by, bx);
                    let dot = _mm256_madd_epi16(ones, _mm256_maddubs_epi16(ax, sy));
                    let d = _mm256_set1_ps(xd * f16::from_bits(y.d).to_f32());
                    *acc = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(dot), *acc);
                }
            }
            for (lane, acc) in acc.iter().enumerate() {
                *out.get_unchecked_mut(base + lane) = hsum_f32_8(*acc);
            }
        }
    }

    macro_rules! tiled_entry {
        ($name:ident, $tile:ident, $weight:ty, $input:ty) => {
            /// # Safety
            ///
            /// AVX2 and FMA must be available, `out.len() == rows`, and
            /// `inputs` must hold `rows * weight.len()` blocks.
            #[target_feature(enable = "avx2,fma")]
            pub(super) unsafe fn $name(
                weight: &[$weight],
                inputs: &[$input],
                rows: usize,
                out: &mut [f32],
            ) {
                let mut base = 0;
                // SAFETY: every tile stays within `rows`, checked by the loop.
                unsafe {
                    while base + TILE <= rows {
                        $tile::<TILE>(weight, inputs, rows, base, out);
                        base += TILE;
                    }
                    // The remainder is smaller than one tile, so it decodes
                    // each block once more; instantiating every width keeps
                    // the accumulators in registers there too.
                    match rows - base {
                        0 => {}
                        1 => $tile::<1>(weight, inputs, rows, base, out),
                        2 => $tile::<2>(weight, inputs, rows, base, out),
                        3 => $tile::<3>(weight, inputs, rows, base, out),
                        4 => $tile::<4>(weight, inputs, rows, base, out),
                        5 => $tile::<5>(weight, inputs, rows, base, out),
                        6 => $tile::<6>(weight, inputs, rows, base, out),
                        7 => $tile::<7>(weight, inputs, rows, base, out),
                        8 => $tile::<8>(weight, inputs, rows, base, out),
                        9 => $tile::<9>(weight, inputs, rows, base, out),
                        10 => $tile::<10>(weight, inputs, rows, base, out),
                        11 => $tile::<11>(weight, inputs, rows, base, out),
                        12 => $tile::<12>(weight, inputs, rows, base, out),
                        13 => $tile::<13>(weight, inputs, rows, base, out),
                        14 => $tile::<14>(weight, inputs, rows, base, out),
                        _ => $tile::<15>(weight, inputs, rows, base, out),
                    }
                }
            }
        };
    }

    tiled_entry!(q4k_row, q4k_tile, Q4KBlock, Q8KBlock);
    tiled_entry!(q6k_row, q6k_tile, Q6KBlock, Q8KBlock);
    tiled_entry!(q80_row, q80_tile, Q80Block, Q80Block);
}

/// Dispatch table entry: dot one compressed weight row against `rows`
/// pre-quantized input rows, writing one f32 per input row.
macro_rules! row_entry {
    ($name:ident, $avx:ident, $weight:ty, $input:ty) => {
        pub(crate) fn $name(weight: &[$weight], inputs: &[$input], rows: usize, out: &mut [f32]) {
            debug_assert_eq!(out.len(), rows);
            debug_assert_eq!(inputs.len(), rows * weight.len());
            #[cfg(target_arch = "x86_64")]
            {
                // SAFETY: `supported` confirms AVX2 and FMA, and the debug
                // assertions above pin the slice lengths the kernel indexes.
                unsafe { avx2::$avx(weight, inputs, rows, out) };
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                unreachable!("the fused kernels require x86_64; callers check `supported`")
            }
        }
    };
}

row_entry!(q4k_q8k_row, q4k_row, Q4KBlock, Q8KBlock);
row_entry!(q6k_q8k_row, q6k_row, Q6KBlock, Q8KBlock);
row_entry!(q80_q80_row, q80_row, Q80Block, Q80Block);
