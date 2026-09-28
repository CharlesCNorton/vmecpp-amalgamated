"""Generate the single-file C++ amalgamation of VMEC++.

Merges the VMEC++ library translation units plus the standalone CLI main() into
one .cc. Project headers (vmecpp/, util/) are inlined once each in dependency
order via a recursive paste-once walk; external includes (Eigen, abseil, HDF5,
NetCDF, nlohmann/json, OpenMP) are left in place, their own include guards
making repetition free. abscab (Apache-2.0) is not header-only, since abscab.hh
holds the declarations and abscab.cc the definitions, so both are inlined.

The FFTX-accelerated transform path (VMECPP_USE_FFTX) is dropped, leaving the
partial-DFT routines VMEC++ falls back to when FFTX is off.

--min-out additionally writes a presentation-stripped layer of the same program:
comments and indentation removed and the per-file SPDX and copyright headers
replaced by a single notice, for reading the whole repository in one pass under a
token budget. The source/header markers are kept so a region maps back to the
commented layer.

--prs-out additionally writes a digest of every VMEC++ pull request that is open
or was closed without merging, read from the GitHub API at the time of the run:
title, description, files changed and the whole conversation, without diffs. It
needs a token in GITHUB_TOKEN or GH_TOKEN, or a logged-in gh.

Usage:
  python amalgamate.py \
      --cpp-root   path/to/vmecpp/src/vmecpp/cpp \
      --abscab-root path/to/abscab-cpp \
      --out        vmecpp_amalgamated.cc \
      --min-out    vmecpp_amalgamated.min.cc \
      --prs-out    vmecpp_unmerged_prs.txt

--abscab-root must contain abscab/abscab.hh and abscab/abscab.cc. Get the
sources VMEC++ pins with:
  git clone https://github.com/proximafusion/vmecpp
  git clone https://github.com/jonathanschilling/abscab-cpp \
      && git -C abscab-cpp checkout 5cfa473b90aab06d7f70d986da0c46c46c1ebe9c
"""

import argparse
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

INC_RE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*([<"])([^>"]+)[>"]')
MARKER_RE = re.compile(r'^//\s*(?:source|header):\s*\S+$')

GITHUB_API = "https://api.github.com"
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
CODEX_ABOUT_RE = re.compile(
    r"<details>\s*<summary>[^<]*About Codex in GitHub.*?</details>", re.S)
BENCH_ROW_RE = re.compile(
    r"^\| `([^`]+)` \| `([^`]+)` (\S+)[^|]*\| `([^`]+)` \S+[^|]*"
    r"\| `([^`]+)` \|$", re.M)
CLANG_TIDY_RE = re.compile(r"(?:warning|error): [^\n]*\]\n```")
CPP_PREFIX = "src/vmecpp/cpp/"
MAX_FILES = 50

# Library TUs relative to the cpp root, mirroring the vmecpp_sources list in
# upstream's CMakeLists; vmec_standalone (the sole main()) last.
# fft_toroidal.cc is listed for parity with upstream and strips to nothing here,
# its whole body sitting behind VMECPP_USE_FFTX. The two Enzyme translation
# units (exact_force_{jvp,vjp}.cc) are excluded: upstream builds them only under
# VMECPP_ENABLE_ENZYME, with a Clang/Enzyme plugin, and their uses in
# ideal_mhd_model.cc sit behind that same define.
TUS = [
    "util/file_io/file_io.cc",
    "util/hdf5_io/hdf5_io.cc",
    "util/json_io/json_io.cc",
    "util/netcdf_io/netcdf_io.cc",
    "util/testing/numerical_comparison_lib.cc",
    "vmecpp/common/composed_types_lib/composed_types_lib.cc",
    "vmecpp/common/flow_control/flow_control.cc",
    "vmecpp/common/fourier_basis/fourier_basis.cc",
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
    "vmecpp/vmec/geometry/geometry.cc",
    "vmecpp/vmec/geometry/vmec_geometry.cc",
    "vmecpp/vmec/handover_storage/handover_storage.cc",
    "vmecpp/vmec/ideal_mhd_model/dft_toroidal.cc",
    "vmecpp/vmec/ideal_mhd_model/fft_toroidal.cc",
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

    Match the inline_roots prefixes first (covers both <...> and "..." since
    the project uses both). For quoted includes, also try the including file's
    own directory (standard C++ semantics) and accept it only when it lands
    under an inline root, which is how abscab.cc's `#include "abscab.hh"` is
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


def strip_presentation(text):
    """Remove comments and indentation, keeping the source/header markers.

    String and character literals are scanned, so a // or /* inside one survives.
    A block comment becomes one space, which cannot weld two tokens together; a
    line comment leaves its newline, so nothing else moves onto another line.
    Runs of blank lines collapse to one, and the trailing backslash of a macro
    continuation is kept."""
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c in '"\'':
            j = i + 1
            while j < n:
                if text[j] == '\\':
                    j += 2
                    continue
                if text[j] == c:
                    j += 1
                    break
                if text[j] == '\n':
                    break
                j += 1
            out.append(text[i:j])
            i = j
        elif text.startswith('//', i):
            j = text.find('\n', i)
            j = n if j < 0 else j
            comment = text[i:j].strip()
            if MARKER_RE.match(comment):
                out.append(comment)
            i = j
        elif text.startswith('/*', i):
            j = text.find('*/', i + 2)
            j = n if j < 0 else j + 2
            out.append(' ')
            i = j
        else:
            out.append(c)
            i += 1
    kept = []
    for line in "".join(out).split("\n"):
        line = line.strip()
        if line or (kept and kept[-1]):
            kept.append(line)
    return "\n".join(kept).strip("\n") + "\n"


def git(cpp, *args):
    try:
        return subprocess.run(["git", "-C", str(cpp), *args],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def github_token():
    for var in ("GITHUB_TOKEN", "GH_TOKEN"):
        if os.environ.get(var):
            return os.environ[var]
    try:
        return subprocess.run(["gh", "auth", "token"], capture_output=True,
                              text=True).stdout.strip()
    except OSError:
        return ""


def github_get(url, token):
    headers = {"Accept": "application/vnd.github+json",
               "Authorization": f"Bearer {token}",
               "X-GitHub-Api-Version": "2022-11-28",
               "User-Agent": "vmecpp-amalgamate"}
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r), r.headers.get("Link", "")
        except urllib.error.HTTPError as e:
            if e.code not in (403, 429, 500, 502, 503, 504) or attempt == 5:
                raise
            wait = int(e.headers.get("Retry-After") or 0) or 5 * 2 ** attempt
        except urllib.error.URLError:
            if attempt == 5:
                raise
            wait = 5 * 2 ** attempt
        time.sleep(wait)


def github_all(path, token):
    """Every item of a paginated list endpoint."""
    url = f"{GITHUB_API}{path}{'&' if '?' in path else '?'}per_page=100"
    items = []
    while url:
        data, link = github_get(url, token)
        items.extend(data)
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = m.group(1) if m else None
    return items


def graphite_stack(body):
    """A Graphite stack notice reduced to its stack, top first, else None."""
    if "This stack of pull requests is managed by" not in body:
        return None
    stack = []
    for line in body.split("\n"):
        m = re.match(r"\* \*\*#(\d+)\*\*", line)
        if m:
            item = f"#{m.group(1)}"
            if "\U0001f448" in line:
                item += " (this PR)"
            above = re.findall(r"\[#(\d+)\]\(", line)
            if above:
                item += " (also under " + ", ".join(f"#{n}" for n in above) + ")"
            stack.append(item)
            continue
        m = re.match(r"\* `([^`]+)`", line)
        if m:
            stack.append(m.group(1))
    return "Graphite stack, top first: " + " > ".join(stack) if stack else None


def benchmark_alert(body):
    """A github-action-benchmark alert reduced to its measurements, else None."""
    if "github-action-benchmark" not in body:
        return None
    rows = BENCH_ROW_RE.findall(body)
    if not rows:
        return None

    def num(s):
        try:
            return f"{float(s):.3g}"
        except ValueError:
            return s

    head = "Benchmark alert"
    m = re.search(r"threshold `([^`]+)`", body)
    if m:
        head += f", threshold {m.group(1)}"
    m = re.search(r"Current: (\w+) \| Previous: (\w+)", body)
    if m:
        head += f", {m.group(1)[:8]} against {m.group(2)[:8]}"
    return head + ": " + "; ".join(
        f"{name.split('::')[-1]} {num(cur)} vs {num(prev)} {unit} (ratio {ratio})"
        for name, cur, unit, prev, ratio in rows)


def clang_tidy(body):
    """A clang-tidy diagnostic reduced to its message lines, else None."""
    if not CLANG_TIDY_RE.match(body):
        return None
    text = re.sub(r"```.*?```", "", body, flags=re.S)
    text = re.sub(r"\[([^\]\[]+)\]\(https?://[^)\s]*\)", r"\1", text)
    return "\n".join(line for line in text.split("\n") if line.strip())


def pr_text(body):
    """A post as GitHub displays it, less HTML comments and bot boilerplate."""
    body = (body or "").replace("\r\n", "\n").replace("\r", "\n")
    for reduce in (graphite_stack, benchmark_alert, clang_tidy):
        short = reduce(body)
        if short:
            return short
    body = CODEX_ABOUT_RE.sub("", HTML_COMMENT_RE.sub("", body))
    return re.sub(r"\n{3,}", "\n\n", body).strip()


def when(stamp):
    return f"{stamp[:10]} {stamp[11:16]} UTC" if stamp else "?"


def login(user):
    return user["login"] if user else "ghost"


def comment_place(c):
    line, start, outdated = c.get("line"), c.get("start_line"), False
    if line is None and c.get("original_line") is not None:
        line, start, outdated = (c["original_line"],
                                 c.get("original_start_line"), True)
    place = c["path"].removeprefix(CPP_PREFIX)
    if line is not None:
        place += f":{start}-{line}" if start and start != line else f":{line}"
    return place + (" (outdated)" if outdated else "")


def file_change(f):
    name = f["filename"].removeprefix(CPP_PREFIX)
    if f["status"] == "renamed":
        old = f.get("previous_filename", "?").removeprefix(CPP_PREFIX)
        name = f"{old} -> {name}"
    elif f["status"] in ("added", "removed"):
        name += f" ({f['status']})"
    return f"{name} +{f['additions']} -{f['deletions']}"


def render_pr(p, files, comments, reviews, inline):
    """One pull request: header, files changed, description, then every
    comment, review and inline review thread in time order. Pending reviews,
    which only their author can see, are left out."""
    state = "OPEN" if p["state"] == "open" else "CLOSED, not merged"
    if p.get("draft"):
        state += ", draft"
    out = ["#" * 80, f"PR #{p['number']} [{state}] {p['title'].strip()}",
           "#" * 80]
    dates = f"author: {login(p['user'])} | opened: {when(p['created_at'])}"
    if p.get("closed_at"):
        dates += f" | closed: {when(p['closed_at'])}"
    out.append(dates)
    head = p["head"].get("label") or p["head"]["ref"]
    out.append(f"branch: {head} -> {p['base']['ref']}")
    labels = ", ".join(label["name"] for label in p.get("labels") or [])
    if labels:
        out.append(f"labels: {labels}")
    out.append(f"url: {p['html_url']}")
    listed = ", ".join(file_change(f) for f in files[:MAX_FILES])
    if len(files) > MAX_FILES:
        listed += f", and {len(files) - MAX_FILES} more"
    out.append(f"files ({len(files)}): {listed or 'none'}")
    out += ["", "--- description ---", pr_text(p["body"]) or "(empty)", ""]

    events = []
    for c in comments:
        events.append((c["created_at"], 0,
                       f"--- comment by {login(c['user'])}, "
                       f"{when(c['created_at'])} ---", pr_text(c["body"])))
    pending = {r["id"] for r in reviews if r["state"] == "PENDING"}
    for r in reviews:
        body = pr_text(r["body"])
        if r["state"] == "PENDING" or (r["state"] == "COMMENTED" and not body):
            continue
        events.append((r["submitted_at"] or "", 1,
                       f"--- review by {login(r['user'])}, "
                       f"{when(r['submitted_at'])}: {r['state']} ---", body))
    inline = [c for c in inline
              if c.get("pull_request_review_id") not in pending]
    by_id = {c["id"]: c for c in inline}
    threads = {}
    for c in inline:
        root, seen = c, set()
        while root.get("in_reply_to_id") in by_id and root["id"] not in seen:
            seen.add(root["id"])
            root = by_id[root["in_reply_to_id"]]
        threads.setdefault(root["id"], []).append(c)
    for thread in threads.values():
        thread.sort(key=lambda c: c["created_at"])
        posts = [f"> {login(c['user'])}, {when(c['created_at'])}:\n"
                 f"{pr_text(c['body'])}" for c in thread]
        events.append((thread[0]["created_at"], 2,
                       f"--- inline thread on {comment_place(thread[0])} ---",
                       "\n\n".join(posts)))
    for _, _, heading, body in sorted(events, key=lambda e: e[:2]):
        out.append(heading)
        if body:
            out.append(body)
        out.append("")
    return "\n".join(out) + "\n"


def write_pr_digest(path, repo, token, prov):
    prs = [p for p in github_all(f"/repos/{repo}/pulls?state=all", token)
           if p["state"] == "open" or p["merged_at"] is None]
    prs.sort(key=lambda p: (p["state"] != "open", p["number"]))

    def fetch(p):
        n = p["number"]
        return render_pr(p, *(github_all(f"/repos/{repo}/{kind}/{n}/{what}", token)
                              for kind, what in (("pulls", "files"),
                                                 ("issues", "comments"),
                                                 ("pulls", "reviews"),
                                                 ("pulls", "comments"))))

    with ThreadPoolExecutor(6) as ex:
        rendered = list(ex.map(fetch, prs))
    n_open = sum(p["state"] == "open" for p in prs)
    taken = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    header = f"""VMEC++ pull requests that are open or were closed without merging, from
github.com/{repo}, taken {taken}: {n_open} open, {len(prs) - n_open} closed. The
amalgamation beside this file is built from {prov}; the changes of
merged pull requests are in it, and those pull requests are not listed here.

Each entry gives the pull request's state, author, dates, branches, labels and
files changed, then its description and its conversation in time order:
comments, review verdicts and summaries, and inline review comments grouped by
thread under the file and line they address. Paths under {CPP_PREFIX} are
given relative to it, as the amalgamation's source and header markers give
them. Diffs are not included. HTML comments are removed, a
Graphite stack notice is reduced to the stack, a benchmark alert to its
measurements, and a clang-tidy diagnostic to its message. Open pull requests
come first, then closed ones, each in number order.

"""
    path.write_text(header + "\n".join(rendered), encoding="utf-8", newline="\n")
    return n_open, len(prs) - n_open


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cpp-root", default="vmecpp/src/vmecpp/cpp", type=Path,
                    help="path to the VMEC++ cpp source root")
    ap.add_argument("--abscab-root", default="abscab-cpp", type=Path,
                    help="path containing abscab/abscab.{hh,cc}")
    ap.add_argument("--out", default="vmecpp_amalgamated.cc", type=Path,
                    help="output .cc path")
    ap.add_argument("--min-out", default=None, type=Path,
                    help="also write the presentation-stripped layer here")
    ap.add_argument("--provenance", default="",
                    help="provenance string for the banner; default uses git")
    ap.add_argument("--prs-out", default=None, type=Path,
                    help="also write the digest of unmerged pull requests here")
    ap.add_argument("--prs-repo", default="proximafusion/vmecpp",
                    help="GitHub repository the digest reads")
    args = ap.parse_args()

    cpp = args.cpp_root.resolve()
    abscab_root = args.abscab_root.resolve()
    if not (cpp / "vmecpp" / "vmec" / "vmec" / "vmec.cc").is_file():
        ap.error(f"--cpp-root does not look like a VMEC++ cpp root: {cpp}")
    token = github_token() if args.prs_out else ""
    if args.prs_out and not token:
        ap.error("--prs-out needs a GitHub token in GITHUB_TOKEN or GH_TOKEN, "
                 "or a logged-in gh")

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
// complete output suite). Two paths upstream keeps behind build defines are
// left out: the FFTX/SPIRAL toroidal transform (VMECPP_USE_FFTX), leaving the
// partial-DFT routines VMEC++ falls back to when it is off, and the two Enzyme
// autodiff translation units (VMECPP_ENABLE_ENZYME), which compile only under a
// Clang/Enzyme plugin. Tests, benchmarks, mockups, the makegrid CLI and the
// pybind module are not included.
//
// Build-time dependencies, pinned to what VMEC++ fetches: Eigen 5.0.1,
// abseil-cpp 20260107.1 (must provide absl/log), nlohmann/json 3.11.3, HDF5
// (C++ API), NetCDF-C, OpenMP{', abscab @ 5cfa473b (inlined above)' if abscab_inlined else ''}.
// Build flags mirror VMEC++'s Release build: -O3 -DNDEBUG -fno-math-errno with
// EIGEN_DONT_PARALLELIZE and EIGEN_MAX_ALIGN_BYTES pinned to 32. The provided
// CMakeLists.txt fetches the pinned dependencies and builds this file
// directly.
//
// Run:
//   ./vmecpp input.json [n_threads]   # writes input.out.h5
// ============================================================================
"""

    body = "\n".join(chunks)
    args.out.write_text(banner + body + "\n", encoding="utf-8")

    if args.min_out:
        min_banner = f"""// ============================================================================
// VMEC++ - single-file C++ amalgamation, presentation-stripped layer
//
// The same program as {args.out.name}, with comments and indentation
// removed and the per-file SPDX and copyright headers replaced by the single
// notice below. Read this layer to take in the whole of VMEC++ at once; the
// source: and header: markers are kept, so any region maps back to that file,
// where the comments on its routines are.
//
// SPDX-License-Identifier: MIT AND Apache-2.0
//
// VMEC++ (github.com/proximafusion/vmecpp): MIT License, Copyright (c)
// 2024-present Proxima Fusion GmbH. abscab
// (github.com/jonathanschilling/abscab-cpp), the Biot-Savart routines the
// free-boundary path uses: Apache License 2.0, Copyright (c) Jonathan
// Schilling. See LICENSE, NOTICE and THIRD_PARTY_LICENSES/.
//
// Unofficial redistribution; not affiliated with or endorsed by Proxima Fusion.
//
// Provenance: {prov}
//
// Scope matches {args.out.name} exactly: the whole solver, fixed and
// free boundary, every profile parameterization, the complete output suite and
// the standalone main(), less the FFTX/SPIRAL transform (VMECPP_USE_FFTX) and
// the Enzyme autodiff translation units (VMECPP_ENABLE_ENZYME).
//
// Build:
//   cmake --build build --target vmecpp_min
//   ./build/vmecpp_min input.json [n_threads]   # writes input.out.h5
// ============================================================================
"""
        args.min_out.write_text(min_banner + strip_presentation(body),
                                encoding="utf-8")

    if args.prs_out:
        n_open, n_closed = write_pr_digest(args.prs_out, args.prs_repo, token,
                                           prov)

    n_lines = (banner + body).count("\n") + 1
    print(f"wrote {args.out}")
    print(f"  translation units merged : {n_tus}")
    print(f"  project headers inlined  : {len(emitted) - n_tus}")
    print(f"  output lines             : {n_lines}")
    print(f"  output size              : {args.out.stat().st_size / 1024:.0f} KiB")
    print(f"  abscab inlined           : {abscab_inlined}")
    if args.min_out:
        min_text = args.min_out.read_text(encoding="utf-8")
        print(f"  stripped layer           : {args.min_out}")
        print(f"    lines                  : {min_text.count(chr(10))}")
        print(f"    size                   : {args.min_out.stat().st_size / 1024:.0f} KiB"
              f"  ({args.min_out.stat().st_size / args.out.stat().st_size:.1%})")
    if args.prs_out:
        print(f"  PR digest                : {args.prs_out}")
        print(f"    open / closed unmerged : {n_open} / {n_closed}")
        print(f"    size                   : {args.prs_out.stat().st_size / 1024:.0f} KiB")
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
