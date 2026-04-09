//! Pre-computed Lloyd-Max codebook data tables.
//!
//! Static centroid and boundary arrays for common `(bits, dim)` combinations.
// qual:allow(srp) — pure data module: const arrays have no logic to split further

use super::Codebook;

/// Static description of a pre-computed codebook stored as `const` arrays.
pub(super) struct StaticCodebook {
    pub(super) centroids: &'static [f64],
    pub(super) boundaries: &'static [f64],
}

impl StaticCodebook {
    /// Convert to an owned [`Codebook`].
    pub(super) fn to_codebook(&self) -> Codebook {
        Codebook {
            centroids: self.centroids.to_vec(),
            boundaries: self.boundaries.to_vec(),
        }
    }
}

// ---------------------------------------------------------------------------
// Pre-computed codebook statics
// ---------------------------------------------------------------------------

// --- 2-bit codebooks (4 centroids, 3 boundaries) ---

pub(super) static CODEBOOK_2BIT_D32: StaticCodebook = StaticCodebook {
    centroids: &CENTROIDS_2B_D32,
    boundaries: &BOUNDARIES_2B_D32,
};
pub(super) static CODEBOOK_2BIT_D64: StaticCodebook = StaticCodebook {
    centroids: &CENTROIDS_2B_D64,
    boundaries: &BOUNDARIES_2B_D64,
};
pub(super) static CODEBOOK_2BIT_D128: StaticCodebook = StaticCodebook {
    centroids: &CENTROIDS_2B_D128,
    boundaries: &BOUNDARIES_2B_D128,
};
pub(super) static CODEBOOK_2BIT_D256: StaticCodebook = StaticCodebook {
    centroids: &CENTROIDS_2B_D256,
    boundaries: &BOUNDARIES_2B_D256,
};

// --- 3-bit codebooks (8 centroids, 7 boundaries) ---

pub(super) static CODEBOOK_3BIT_D32: StaticCodebook = StaticCodebook {
    centroids: &CENTROIDS_3B_D32,
    boundaries: &BOUNDARIES_3B_D32,
};
pub(super) static CODEBOOK_3BIT_D64: StaticCodebook = StaticCodebook {
    centroids: &CENTROIDS_3B_D64,
    boundaries: &BOUNDARIES_3B_D64,
};
pub(super) static CODEBOOK_3BIT_D128: StaticCodebook = StaticCodebook {
    centroids: &CENTROIDS_3B_D128,
    boundaries: &BOUNDARIES_3B_D128,
};
pub(super) static CODEBOOK_3BIT_D256: StaticCodebook = StaticCodebook {
    centroids: &CENTROIDS_3B_D256,
    boundaries: &BOUNDARIES_3B_D256,
};

// --- 4-bit codebooks (16 centroids, 15 boundaries) ---

pub(super) static CODEBOOK_4BIT_D32: StaticCodebook = StaticCodebook {
    centroids: &CENTROIDS_4B_D32,
    boundaries: &BOUNDARIES_4B_D32,
};
pub(super) static CODEBOOK_4BIT_D64: StaticCodebook = StaticCodebook {
    centroids: &CENTROIDS_4B_D64,
    boundaries: &BOUNDARIES_4B_D64,
};
pub(super) static CODEBOOK_4BIT_D128: StaticCodebook = StaticCodebook {
    centroids: &CENTROIDS_4B_D128,
    boundaries: &BOUNDARIES_4B_D128,
};
pub(super) static CODEBOOK_4BIT_D256: StaticCodebook = StaticCodebook {
    centroids: &CENTROIDS_4B_D256,
    boundaries: &BOUNDARIES_4B_D256,
};

/// Look up a pre-computed static codebook reference for the given
/// `(bits, dim)` pair, returning `None` if not found.
///
/// Pure Operation: only match logic, returns a reference.
pub(super) fn lookup_static_codebook_ref(bits: u8, dim: usize) -> Option<&'static StaticCodebook> {
    match (bits, dim) {
        (2, 32) => Some(&CODEBOOK_2BIT_D32),
        (2, 64) => Some(&CODEBOOK_2BIT_D64),
        (2, 128) => Some(&CODEBOOK_2BIT_D128),
        (2, 256) => Some(&CODEBOOK_2BIT_D256),
        (3, 32) => Some(&CODEBOOK_3BIT_D32),
        (3, 64) => Some(&CODEBOOK_3BIT_D64),
        (3, 128) => Some(&CODEBOOK_3BIT_D128),
        (3, 256) => Some(&CODEBOOK_3BIT_D256),
        (4, 32) => Some(&CODEBOOK_4BIT_D32),
        (4, 64) => Some(&CODEBOOK_4BIT_D64),
        (4, 128) => Some(&CODEBOOK_4BIT_D128),
        (4, 256) => Some(&CODEBOOK_4BIT_D256),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Pre-computed tables
// ---------------------------------------------------------------------------

// 2-bit, d=32
const CENTROIDS_2B_D32: [f64; 4] = [
    -0.2633194690486469,
    -0.07980195923950476,
    0.07980195923950384,
    0.2633194690486451,
];
const BOUNDARIES_2B_D32: [f64; 3] = [-0.1715607141440758, 0.0, 0.17156071414407448];

// 3-bit, d=32
const CENTROIDS_3B_D32: [f64; 8] = [
    -0.3662684197149945,
    -0.2324607608635176,
    -0.13175624361004853,
    -0.042851570066700616,
    0.04285157006670112,
    0.1317562436100487,
    0.23246076086351877,
    0.36626841971499513,
];
const BOUNDARIES_3B_D32: [f64; 7] = [
    -0.29936459028925605,
    -0.18210850223678307,
    -0.08730390683837458,
    0.0,
    0.08730390683837491,
    0.18210850223678374,
    0.29936459028925694,
];

// 4-bit, d=32
const CENTROIDS_4B_D32: [f64; 16] = [
    -0.45356484406403885,
    -0.3500919292405678,
    -0.276742371576412,
    -0.2163621312970215,
    -0.16307095286109716,
    -0.11401417528619187,
    -0.06749924527967034,
    -0.02235501645565892,
    0.022355016455658717,
    0.06749924527967022,
    0.11401417528619151,
    0.16307095286109707,
    0.21636213129702167,
    0.27674237157641146,
    0.35009192924056776,
    0.4535648440640396,
];
const BOUNDARIES_4B_D32: [f64; 15] = [
    -0.40182838665230336,
    -0.31341715040848994,
    -0.24655225143671677,
    -0.18971654207905933,
    -0.13854256407364451,
    -0.09075671028293111,
    -0.04492713086766463,
    0.0,
    0.044927130867664464,
    0.09075671028293086,
    0.1385425640736443,
    0.18971654207905936,
    0.24655225143671655,
    0.3134171504084896,
    0.4018283866523037,
];

// 2-bit, d=64
const CENTROIDS_2B_D64: [f64; 4] = [
    -0.18749689292196112,
    -0.05651489047635318,
    0.05651489047635313,
    0.18749689292196103,
];
const BOUNDARIES_2B_D64: [f64; 3] = [-0.12200589169915715, 0.0, 0.12200589169915708];

// 2-bit, d=128
const CENTROIDS_2B_D128: [f64; 4] = [
    -0.13304154846077318,
    -0.03999160906335877,
    0.03999160906335891,
    0.13304154846077348,
];
const BOUNDARIES_2B_D128: [f64; 3] = [-0.08651657876206598, 0.0, 0.086_516_578_762_066_2];

// 2-bit, d=256
const CENTROIDS_2B_D256: [f64; 4] = [
    -0.09423779913129633,
    -0.02828860721372146,
    0.02828860721372166,
    0.09423779913129664,
];
const BOUNDARIES_2B_D256: [f64; 3] = [-0.061_263_203_172_508_9, 0.0, 0.06126320317250915];

// 3-bit, d=64
const CENTROIDS_3B_D64: [f64; 8] = [
    -0.26391407457137683,
    -0.16616801009118487,
    -0.093_832_375_844_160_5,
    -0.03046922045737837,
    0.03046922045737837,
    0.093_832_375_844_160_5,
    0.16616801009118487,
    0.26391407457137683,
];
const BOUNDARIES_3B_D64: [f64; 7] = [
    -0.21504104233128085,
    -0.13000019296767268,
    -0.06215079815076943,
    0.0,
    0.06215079815076943,
    0.13000019296767268,
    0.21504104233128085,
];

// 3-bit, d=128
const CENTROIDS_3B_D128: [f64; 8] = [
    -0.18839728518004373,
    -0.11813986946554235,
    -0.06658568378325364,
    -0.02160433847349997,
    0.02160433847349997,
    0.06658568378325364,
    0.11813986946554235,
    0.18839728518004373,
];
const BOUNDARIES_3B_D128: [f64; 7] = [
    -0.15326857732279303,
    -0.09236277662439799,
    -0.044_095_011_128_376_8,
    0.0,
    0.044_095_011_128_376_8,
    0.09236277662439799,
    0.15326857732279303,
];

// 3-bit, d=256
const CENTROIDS_3B_D256: [f64; 8] = [
    -0.13385436276083063,
    -0.083_765_531_459_768_9,
    -0.04716676527922715,
    -0.01529750782483941,
    0.01529750782483941,
    0.04716676527922715,
    0.083_765_531_459_768_9,
    0.13385436276083063,
];
const BOUNDARIES_3B_D256: [f64; 7] = [
    -0.10880994711029976,
    -0.06546614836949802,
    -0.03123213655203328,
    0.0,
    0.03123213655203328,
    0.06546614836949802,
    0.10880994711029976,
];

// 4-bit, d=64
const CENTROIDS_4B_D64: [f64; 16] = [
    -0.33092994168409773,
    -0.25307088610074774,
    -0.19901983361887085,
    -0.15508179062990365,
    -0.11662310388676207,
    -0.08141753279040376,
    -0.04815672368589858,
    -0.015_941_930_352_081_4,
    0.015_941_930_352_081_4,
    0.04815672368589858,
    0.08141753279040376,
    0.11662310388676207,
    0.15508179062990365,
    0.19901983361887085,
    0.25307088610074774,
    0.33092994168409773,
];
const BOUNDARIES_4B_D64: [f64; 15] = [
    -0.292_000_413_892_422_7,
    -0.22604535985980928,
    -0.17705081212438725,
    -0.13585244725833287,
    -0.09902031833858291,
    -0.06478712823815116,
    -0.03204932701898999,
    0.0,
    0.03204932701898999,
    0.06478712823815116,
    0.09902031833858291,
    0.13585244725833287,
    0.17705081212438725,
    0.22604535985980928,
    0.292_000_413_892_422_7,
];

// 4-bit, d=128
const CENTROIDS_4B_D128: [f64; 16] = [
    -0.23777655506958537,
    -0.18096588552769086,
    -0.14193912272806147,
    -0.11041538921898804,
    -0.08293881469006784,
    -0.05785765497830671,
    -0.03420549908335103,
    -0.01132093590150223,
    0.01132093590150223,
    0.03420549908335103,
    0.05785765497830671,
    0.08293881469006784,
    0.11041538921898804,
    0.14193912272806147,
    0.18096588552769086,
    0.23777655506958537,
];
const BOUNDARIES_4B_D128: [f64; 15] = [
    -0.209_371_220_298_638_1,
    -0.16145250412787615,
    -0.12617725597352475,
    -0.09667710195452794,
    -0.07039823483418728,
    -0.04603157703082887,
    -0.02276321749242663,
    0.0,
    0.02276321749242663,
    0.04603157703082887,
    0.07039823483418728,
    0.09667710195452794,
    0.12617725597352475,
    0.16145250412787615,
    0.209_371_220_298_638_1,
];

// 4-bit, d=256
const CENTROIDS_4B_D256: [f64; 16] = [
    -0.16949853314441155,
    -0.12868871755030106,
    -0.10080108457584613,
    -0.07834675699488723,
    -0.05881658417438018,
    -0.04101444098641885,
    -0.02424206232116148,
    -0.00802245010411462,
    0.00802245010411462,
    0.02424206232116148,
    0.04101444098641885,
    0.05881658417438018,
    0.07834675699488723,
    0.10080108457584613,
    0.12868871755030106,
    0.16949853314441155,
];
const BOUNDARIES_4B_D256: [f64; 15] = [
    -0.14909362534735632,
    -0.114_744_901_063_073_6,
    -0.08957392078536669,
    -0.06858167058463371,
    -0.04991551258039952,
    -0.03262825165379016,
    -0.01613225621263805,
    0.0,
    0.01613225621263805,
    0.03262825165379016,
    0.04991551258039952,
    0.06858167058463371,
    0.08957392078536669,
    0.114_744_901_063_073_6,
    0.14909362534735632,
];
