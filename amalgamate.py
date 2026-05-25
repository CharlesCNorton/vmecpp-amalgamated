"""Generate the single-file C++ amalgamation of VMEC++.

Merges the VMEC++ library translation units plus the standalone CLI main() into
one .cc. Project headers (vmecpp/, util/) are inlined once each in dependency
order via a recursive paste-once walk; external includes (Eigen, abseil, HDF5,
NetCDF, nlohmann/json, OpenMP) are left in place, their own include guards
making repetition free. abscab (Apache-2.0) is not header-only -- abscab.hh
holds declarations and abscab.cc the definitions -- so both are inlined.

The FFTX-accelerated transform path (VMECPP_USE_FFTX) is dropped, keeping the
byte-identical partial-DFT path VMEC++ uses by default.

Usage:
  python amalgamate.py \
      --cpp-root   path/to/vmecpp/src/vmecpp/cpp \
      --abscab-root path/to/abscab-cpp \
      --out        vmecpp_amalgamated.cc

--abscab-root must contain abscab/abscab.hh and abscab/abscab.cc. Get the
sources VMEC++ pins with:
  git clone https://github.com/proximafusion/vmecpp
  git clone https://github.com/jonathanschilling/abscab-cpp \
      && git -C abscab-cpp checkout 5cfa473b90aab06d7f70d986da0c46c46c1ebe9c
"""

import argparse
import re
import subprocess
from pathlib import Path

INC_RE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*([<"])([^>"]+)[>"]')

# Library TUs relative to the cpp root; vmec_standalone (the sole main()) last.
TUS = [
    "util/file_io/file_io.cc",
    "util/hdf5_io/hdf5_io.cc",
    "util/json_io/json_io.cc",
    "util/netcdf_io/netcdf_io.cc",
    "util/testing/numerical_comparison_lib.cc",
    "vmecpp/common/composed_types_lib/composed_types_lib.cc",
    "vmecpp/common/flow_control/flow_control.cc",
    "vmecpp/common/fourier_basis_fast_poloidal/fourier_basis_fast_poloidal.cc",
    "vmecpp/common/fourier_basis_fast_toroidal/fourier_basis_fast_toroidal.cc",
    "vmecpp/common/magnetic_configuration_lib/magnetic_configuration_lib.cc",
    "vmecpp/common/magnetic_field_provider/magnetic_field_provider_lib.cc",
    "vmecpp/common/makegrid_lib/makegrid_lib.cc",
    "vmecpp/common/sizes/sizes.cc",
    "vmecpp/common/util/util.cc",
    "vmecpp/common/vmec_indata/boundary_from_json.cc",
    "vmecpp/common/vmec_indata/vmec_indata.cc",
    "vmecpp/free_boundary/external_magnetic_field/external_magnetic_field.cc",
    "vmecpp/free_boundary/laplace_solver/laplace_solver.cc",
    "vmecpp/free_boundary/mgrid_provider/mgrid_provider.cc",
    "vmecpp/free_boundary/nestor/nestor.cc",
    "vmecpp/free_boundary/only_coils/only_coils.cc",
    "vmecpp/free_boundary/regularized_integrals/regularized_integrals.cc",
    "vmecpp/free_boundary/singular_integrals/singular_integrals.cc",
    "vmecpp/free_boundary/surface_geometry/surface_geometry.cc",
    "vmecpp/free_boundary/tangential_partitioning/tangential_partitioning.cc",
    "vmecpp/vmec/boundaries/boundaries.cc",
    "vmecpp/vmec/boundaries/guess_magnetic_axis.cc",
    "vmecpp/vmec/fourier_coefficients/fourier_coefficients.cc",
    "vmecpp/vmec/fourier_forces/fourier_forces.cc",
    "vmecpp/vmec/fourier_geometry/fourier_geometry.cc",
    "vmecpp/vmec/fourier_velocity/fourier_velocity.cc",
    "vmecpp/vmec/handover_storage/handover_storage.cc",
    "vmecpp/vmec/ideal_mhd_model/dft_toroidal.cc",
    "vmecpp/vmec/ideal_mhd_model/ideal_mhd_model.cc",
    "vmecpp/vmec/iteration_logger/iteration_logger.cc",
    "vmecpp/vmec/output_quantities/output_quantities.cc",
    "vmecpp/vmec/profile_parameterization_data/profile_parameterization_data.cc",
    "vmecpp/vmec/radial_partitioning/radial_partitioning.cc",
    "vmecpp/vmec/radial_profiles/radial_profiles.cc",
    "vmecpp/vmec/thread_local_storage/thread_local_storage.cc",
    "vmecpp/vmec/vmec/vmec.cc",
    "vmecpp/vmec/vmec_constants/vmec_constants.cc",
    "vmecpp/vmec/vmec_standalone/vmec_standalone.cc",
]


def build_inline_roots(cpp, abscab_root):
    roots = [("vmecpp/", cpp), ("util/", cpp)]
    extra_tus = []
    if (abscab_root / "abscab" / "abscab.hh").is_file():
        roots.append(("abscab/", abscab_root))
        extra_tus.append((abscab_root / "abscab" / "abscab.cc", "abscab/abscab.cc"))
    return roots, extra_tus


def resolve_inline(quote, name, cur_dir, inline_roots):
    """On-disk path if `name` is an include we should inline, else None.

    Match the INLINE_ROOTS prefixes first (covers both <...> and "..." since
    the project uses both). For quoted includes, also try the including file's
    own directory (standard C++ semantics) and accept it only when it lands
    under an inline root -- that is how abscab.cc's `#include "abscab.hh"` is
    found.
    """
    for prefix, root in inline_roots:
        if name.startswith(prefix):
            p = root / name
            return p if p.is_file() else None
    if quote == '"':
        cand = (cur_dir / name).resolve()
        if cand.is_file():
            for _prefix, root in inline_roots:
                try:
                    cand.relative_to(root.resolve())
                    return cand
                except ValueError:
                    continue
    return None


def rel_label(path, roots):
    path = path.resolve()
    for root in roots:
        try:
            return path.relative_to(root.resolve()).as_posix()
        except ValueError:
            continue
    return path.name


def strip_fftx(text):
    """Evaluate VMECPP_USE_FFTX as undefined: drop #ifdef branches, keep #else;
    keep the #ifndef branch. All other preprocessor directives pass through."""
    lines = text.split("\n")
    n = len(lines)
    out = []

    def find_block(start):
        depth = 0
        else_idx = None
        j = start + 1
        while j < n:
            s = lines[j].lstrip()
            if re.match(r'#[ \t]*if', s):
                depth += 1
            elif re.match(r'#[ \t]*endif', s):
                if depth == 0:
                    return else_idx, j
                depth -= 1
            elif re.match(r'#[ \t]*else', s) and depth == 0 and else_idx is None:
                else_idx = j
            j += 1
        return else_idx, n - 1

    i = 0
    while i < n:
        s = lines[i].lstrip()
        if (re.match(r'#[ \t]*ifdef[ \t]+VMECPP_USE_FFTX\b', s)
                or re.match(r'#[ \t]*if[ \t]+defined[ \t]*\([ \t]*'
                            r'VMECPP_USE_FFTX[ \t]*\)', s)):
            else_idx, endif_idx = find_block(i)
            if else_idx is not None:
                out.extend(lines[else_idx + 1:endif_idx])
            i = endif_idx + 1
            continue
        if re.match(r'#[ \t]*ifndef[ \t]+VMECPP_USE_FFTX\b', s):
            else_idx, endif_idx = find_block(i)
            end_then = else_idx if else_idx is not None else endif_idx
            out.extend(lines[i + 1:end_then])
            i = endif_idx + 1
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def git(cpp, *args):
    try:
        return subprocess.run(["git", "-C", str(cpp), *args],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cpp-root", default="vmecpp/src/vmecpp/cpp", type=Path,
                    help="path to the VMEC++ cpp source root")
    ap.add_argument("--abscab-root", default="abscab-cpp", type=Path,
                    help="path containing abscab/abscab.{hh,cc}")
    ap.add_argument("--out", default="vmecpp_amalgamated.cc", type=Path,
                    help="output .cc path")
    ap.add_argument("--provenance", default="",
                    help="provenance string for the banner; default uses git")
    args = ap.parse_args()

    cpp = args.cpp_root.resolve()
    abscab_root = args.abscab_root.resolve()
    if not (cpp / "vmecpp" / "vmec" / "vmec" / "vmec.cc").is_file():
        ap.error(f"--cpp-root does not look like a VMEC++ cpp root: {cpp}")

    inline_roots, extra_tus = build_inline_roots(cpp, abscab_root)
    abscab_inlined = any(p == "abscab/" for p, _ in inline_roots)
    label_roots = [cpp, abscab_root]

    emitted = set()
    chunks = []
    unresolved = []
    external = {}

    def walk(path, rel, kind):
        path = path.resolve()
        if path in emitted:
            return
        emitted.add(path)
        cur_dir = path.parent
        text = strip_fftx(path.read_text(encoding="utf-8", errors="replace"))
        chunks.append(f"\n// {'=' * 76}\n// {kind}: {rel}\n// {'=' * 76}")
        buf = []
        for line in text.split("\n"):
            m = INC_RE.match(line)
            if m:
                quote, name = m.group(1), m.group(2)
                inline_p = resolve_inline(quote, name, cur_dir, inline_roots)
                if inline_p is not None:
                    if buf:
                        chunks.append("\n".join(buf))
                        buf = []
                    walk(inline_p, rel_label(inline_p, label_roots), "header")
                    continue
                if quote == '"' and name.startswith(("vmecpp/", "util/")):
                    unresolved.append((rel, name))
                    buf.append(f"// [amalg] UNRESOLVED: {line.strip()}")
                    continue
                external.setdefault((quote, name), line.strip())
                buf.append(line)
                continue
            buf.append(line)
        if buf:
            chunks.append("\n".join(buf))

    for path, rel in extra_tus:
        walk(path, rel, "source")
    for rel in TUS:
        walk(cpp / rel, rel, "source")

    n_tus = len(TUS) + len(extra_tus)
    if args.provenance:
        prov = args.provenance
    else:
        desc = git(cpp, "describe", "--tags", "--always")
        prov = f"github.com/proximafusion/vmecpp{(' ' + desc) if desc else ''}"

    banner = f"""// ============================================================================
// VMEC++ - single-file C++ amalgamation
//
// A mechanical, paste-once merge of VMEC++ into one translation unit. VMEC++ is
// distributed under the MIT License, Copyright (c) 2024-present Proxima Fusion
// GmbH; its per-file SPDX/copyright headers are preserved inline below. This
// file also inlines abscab (github.com/jonathanschilling/abscab-cpp), the
// Biot-Savart routines used by the free-boundary path, distributed under the
// Apache License 2.0. See LICENSE, NOTICE and THIRD_PARTY_LICENSES/.
//
// Unofficial redistribution; not affiliated with or endorsed by Proxima Fusion.
//
// Provenance: {prov}
//
// Scope: the full solver (fixed + free boundary, all profile parameterizations,
// complete output suite). The FFTX/SPIRAL kernel path (VMECPP_USE_FFTX) is
// omitted; the byte-identical partial-DFT path VMEC++ uses by default is kept.
// Tests, benchmarks, mockups and the pybind module are not included.
//
// Build-time dependencies (pinned to what VMEC++ fetches, so results match
// bit-for-bit): Eigen 5.0.1, abseil-cpp @ 4447c756 (must provide absl/log),
// nlohmann/json 3.11.3, HDF5 (C++ API), NetCDF-C, LAPACK, OpenMP{', abscab @ 5cfa473b (inlined above)' if abscab_inlined else ''}.
// Build flags mirror VMEC++'s Release build: -O3 -DNDEBUG -fno-math-errno with
// EIGEN_DONT_PARALLELIZE. The provided CMakeLists.txt fetches the pinned
// dependencies and builds this file directly.
//
// Run:
//   ./vmecpp input.json [n_threads]   # writes input.out.h5
// ============================================================================
"""

    args.out.write_text(banner + "\n".join(chunks) + "\n", encoding="utf-8")

    n_lines = (banner + "\n".join(chunks)).count("\n") + 1
    print(f"wrote {args.out}")
    print(f"  translation units merged : {n_tus}")
    print(f"  project headers inlined  : {len(emitted) - n_tus}")
    print(f"  output lines             : {n_lines}")
    print(f"  output size              : {args.out.stat().st_size / 1024:.0f} KiB")
    print(f"  abscab inlined           : {abscab_inlined}")
    if unresolved:
        print(f"  UNRESOLVED includes ({len(unresolved)}):")
        for rel, name in unresolved:
            print(f"    {name}  (from {rel})")
    print("  external includes:")
    for (q, name) in sorted(external, key=lambda k: (k[0] == '"', k[1])):
        lb, rb = ('<', '>') if q == '<' else ('"', '"')
        print(f"    {lb}{name}{rb}")


if __name__ == "__main__":
    main()
