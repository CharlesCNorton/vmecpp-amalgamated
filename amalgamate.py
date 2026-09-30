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

--prs-out additionally writes a digest of the VMEC++ pull requests that change
the C++ core (a file under src/vmecpp/cpp/ other than the pybind11 bindings, or
the top-level CMakeLists.txt) and are open, or were opened on or after
--prs-cutoff and closed without merging, read from the GitHub API at the time
of the run: title, description and the conversation less bot posts, without
diffs, and with each code block in a post replaced by a marker line. It needs a
token in GITHUB_TOKEN or GH_TOKEN, or a logged-in gh.

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
import http.client
import json
import os
import re
import subprocess
import textwrap
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path

INC_RE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*([<"])([^>"]+)[>"]')
MARKER_RE = re.compile(r'^//\s*(?:source|header):\s*\S+$')

GITHUB_API = "https://api.github.com"
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
GRAPHITE_NOTICE = "This stack of pull requests is managed by"
TRAILER_RE = re.compile(
    r"^[ \t]*(?:co-authored-by:.*|\S*[ \t]*generated with \[claude code\]"
    r"\([^)]*\)[ \t]*)(?:\n|$)", re.I | re.M)
MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
HTML_IMAGE_RE = re.compile(r"<img\b[^>]*>", re.I)
IMAGE_ALT_RE = re.compile(r"""\balt\s*=\s*(?:"([^"]*)"|'([^']*)')""", re.I)
LEAD_RE = re.compile(r"((?:[ \t]*>)*)([ \t]*)(.*)")
FENCE_RE = re.compile(r"(`{3,}|~{3,})(.*)")
INDENTED_RE = re.compile(r"(?: {4}| {0,3}\t)")
LIST_ITEM_RE = re.compile(r"[ \t]*(?:[-*+]|\d{1,9}[.)])(?:[ \t]|$)")
CODE_MARK = "[code omitted]"
SUGGESTION_MARK = "[suggested change omitted]"
CPP_PREFIX = "src/vmecpp/cpp/"
PYBIND_DIR = "/pybind11/"
PINS_FILE = "CMakeLists.txt"

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
        except (OSError, http.client.HTTPException):
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


def is_bot(post):
    return (post.get("user") or {}).get("type") == "Bot"


def changes_core(paths):
    """Whether a pull request that changes these paths changes the C++ core: a
    file under src/vmecpp/cpp/, its tests and test data included, other than
    the pybind11 bindings, or the top-level CMakeLists.txt that pins the
    dependencies. A pull request for which GitHub lists no changed files counts
    as changing it."""
    return not paths or any(
        (p.startswith(CPP_PREFIX) and PYBIND_DIR not in p) or p == PINS_FILE
        for p in paths)


def image_mark(alt):
    """An image as the text that stands in for it: its alt text, unless that is
    empty or GitHub's default, image, with or without a file extension."""
    alt = " ".join(alt.split())
    if not alt or re.fullmatch(r"image(\.\w+)?", alt, re.I):
        return "[image]"
    return f"[image: {alt}]"


def html_image_mark(tag):
    m = IMAGE_ALT_RE.search(tag)
    return image_mark(next((g for g in m.groups() if g is not None), "")
                      if m else "")


def lead(line):
    """A line's blockquote depth, its indentation inside the quote, and the
    rest of it."""
    quote, space, rest = LEAD_RE.fullmatch(line).groups()
    depth = quote.count(">")
    if depth and space.startswith(" "):
        space = space[1:]
    return depth, len(space.expandtabs(4)), rest


def strip_code(body):
    """Every code block in a post replaced by a marker line.

    A fence closes at a fence of its own character and at least its length at
    the same quote depth, or where the blockquote or list item holding it ends,
    or else at the end of the post, as GitHub renders an unclosed fence. An
    indented block counts as code after a blank line below a paragraph; below a
    list item the same indentation continues the item, as Markdown reads it."""
    lines, fence = [], None
    for line in body.split("\n"):
        depth, indent, rest = lead(line)
        m = FENCE_RE.fullmatch(rest)
        if fence:
            char, length, f_depth, f_indent = fence
            if (m and m.group(1)[0] == char and len(m.group(1)) >= length
                    and not m.group(2).strip() and depth == f_depth):
                fence = None
                continue
            if depth >= f_depth and (indent >= f_indent or not rest.strip()):
                continue
            fence = None
        if m and not (m.group(1)[0] == "`" and "`" in m.group(2)):
            fence = (m.group(1)[0], len(m.group(1)), depth, indent)
            mark = (SUGGESTION_MARK if m.group(2).strip() == "suggestion"
                    else CODE_MARK)
            lines.append(line[:len(line) - len(rest)] + mark)
            continue
        lines.append(line)

    out, para, i = [], "", 0
    while i < len(lines):
        line = lines[i]
        after_blank = not out or not out[-1].strip()
        if (line.strip() and INDENTED_RE.match(line) and after_blank
                and not LIST_ITEM_RE.match(para) and not para[:1].isspace()):
            end = i
            while i < len(lines) and (not lines[i].strip()
                                      or INDENTED_RE.match(lines[i])):
                if lines[i].strip():
                    end = i + 1
                i += 1
            i = end
            out.append(CODE_MARK)
            para = CODE_MARK
            continue
        if line.strip() and after_blank:
            para = line
        out.append(line)
        i += 1
    return "\n".join(out)


def pr_text(body):
    """A post as GitHub displays it, less HTML comments, Co-authored-by and
    Generated with Claude Code trailers and code blocks, and with each image
    given as its alt text."""
    body = (body or "").replace("\r\n", "\n").replace("\r", "\n")
    body = TRAILER_RE.sub("", HTML_COMMENT_RE.sub("", body))
    body = MD_IMAGE_RE.sub(lambda m: image_mark(m.group(1)), body)
    body = HTML_IMAGE_RE.sub(lambda m: html_image_mark(m.group(0)), body)
    return re.sub(r"\n{3,}", "\n\n", strip_code(body)).strip()


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


def render_pr(p, comments, reviews, inline):
    """One pull request: title, description, then every comment, review and
    inline review thread in time order. Posts from bot accounts, Graphite stack
    notices and pending reviews, which only their author can see, are left
    out."""
    state = "OPEN" if p["state"] == "open" else "CLOSED, not merged"
    if p.get("draft"):
        state += ", draft"
    out = [f"=== PR #{p['number']} [{state}] {p['title'].strip()} ===", "",
           f"--- description by {login(p['user'])}, {when(p['created_at'])} ---",
           pr_text(p["body"]) or "(empty)", ""]

    events = []
    for c in comments:
        if is_bot(c) or GRAPHITE_NOTICE in (c["body"] or ""):
            continue
        events.append((c["created_at"], 0,
                       f"--- comment by {login(c['user'])}, "
                       f"{when(c['created_at'])} ---", pr_text(c["body"])))
    pending = {r["id"] for r in reviews if r["state"] == "PENDING"}
    for r in reviews:
        body = pr_text(r["body"])
        if (r["state"] == "PENDING" or is_bot(r)
                or (r["state"] == "COMMENTED" and not body)):
            continue
        events.append((r["submitted_at"] or "", 1,
                       f"--- review by {login(r['user'])}, "
                       f"{when(r['submitted_at'])}: {r['state']} ---", body))
    inline = [c for c in inline
              if c.get("pull_request_review_id") not in pending
              and not is_bot(c)]
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


def write_pr_digest(path, repo, token, prov, cutoff):
    prs = [p for p in github_all(f"/repos/{repo}/pulls?state=all", token)
           if p["state"] == "open"
           or (p["merged_at"] is None and p["created_at"][:10] >= cutoff)]

    def in_core(p):
        files = github_all(f"/repos/{repo}/pulls/{p['number']}/files", token)
        names = [(f["filename"], f.get("previous_filename")) for f in files]
        return changes_core([n for pair in names for n in pair if n])

    with ThreadPoolExecutor(6) as ex:
        core = list(ex.map(in_core, prs))
    n_outside = core.count(False)
    prs = [p for p, keep in zip(prs, core) if keep]
    prs.sort(key=lambda p: (p["state"] != "open", p["number"]))

    def fetch(p):
        n = p["number"]
        return render_pr(p, *(github_all(f"/repos/{repo}/{kind}/{n}/{what}", token)
                              for kind, what in (("issues", "comments"),
                                                 ("pulls", "reviews"),
                                                 ("pulls", "comments"))))

    with ThreadPoolExecutor(6) as ex:
        rendered = list(ex.map(fetch, prs))
    n_open = sum(p["state"] == "open" for p in prs)
    taken = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def nobreak(s):
        return s.replace(" ", "\0")

    header = (
        f"VMEC++ pull requests from github.com/{repo} that change its C++ core "
        f"and are open, or were opened on or after {cutoff} and closed without "
        f"merging, taken {nobreak(taken)}: {n_open} open, "
        f"{len(prs) - n_open} closed. A pull request changes the C++ core when "
        f"it changes a file under {CPP_PREFIX}, its tests and test data "
        f"included, other than the pybind11 bindings, or the top-level "
        f"{PINS_FILE} that pins the dependencies; one for which GitHub lists "
        f"no changed files is listed as well. The amalgamation beside this "
        f"file is built from {prov}; the changes of merged pull requests are "
        f"in it, and those pull requests are not listed here.",
        f"Each entry gives the pull request's title and state, then its "
        f"description and its conversation in time order: comments, review "
        f"verdicts and summaries, and inline review comments grouped by thread "
        f"under the file and line they address, each post under its author and "
        f"date. Paths under {CPP_PREFIX} are given relative to it, as the "
        f"amalgamation's source and header markers give them. Diffs are left "
        f"out, and each code block in a post is replaced by the line "
        f"{nobreak(CODE_MARK)}, or {nobreak(SUGGESTION_MARK)} for a review "
        f"suggestion. Comments and reviews from bot accounts are left out, as "
        f"are Graphite stack notices; a pull request a bot opened keeps its "
        f"description. HTML comments and Co-authored-by and Generated with "
        f"Claude Code trailers are removed, and an image is given as its alt "
        f"text. Open pull requests come first, then closed ones, each in number "
        f"order.")
    header = "\n\n".join(textwrap.fill(p, 80, break_on_hyphens=False)
                         for p in header).replace("\0", " ")
    path.write_text(header + "\n\n\n" + "\n".join(rendered), encoding="utf-8",
                    newline="\n")
    return n_open, len(prs) - n_open, n_outside


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
    ap.add_argument("--prs-cutoff", default="2026-04-25", type=date.fromisoformat,
                    help="leave closed pull requests opened before this date "
                         "(YYYY-MM-DD) out of the digest")
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
        n_open, n_closed, n_outside = write_pr_digest(
            args.prs_out, args.prs_repo, token, prov,
            args.prs_cutoff.isoformat())

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
        print(f"    outside the C++ core   : {n_outside}")
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
