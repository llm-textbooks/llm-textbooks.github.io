#!/usr/bin/env python3
"""Fail-closed structural validation for every published EPUB download."""

from __future__ import annotations

from pathlib import PurePosixPath, Path
from urllib.parse import unquote
import sys
import zipfile
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent.parent
EPUBS = (
    "public/downloads/volume-1-cuda-llm-serving-ko.epub",
    "public/downloads/volume-2-finetuning-mechanisms-ko.epub",
    "content/volume-3/dist/llm-multi-agent-mechanisms-ko-draft.epub",
)
CONTAINER_NS = {"container": "urn:oasis:names:tc:opendocument:xmlns:container"}
OPF_NS = {"opf": "http://www.idpf.org/2007/opf"}


def member_path(base: str, relative: str) -> str:
    return str(PurePosixPath(base).parent.joinpath(unquote(relative)))


def xhtml_ids(archive: zipfile.ZipFile, member: str) -> set[str]:
    root = ET.fromstring(archive.read(member))
    return {node.attrib["id"] for node in root.iter() if "id" in node.attrib}


def validate_epub(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            member_set = set(names)
            if archive.testzip() is not None:
                errors.append(f"CRC failure in {archive.testzip()}")
            if not names or names[0] != "mimetype":
                errors.append("mimetype must be the first ZIP member")
            elif archive.getinfo("mimetype").compress_type != zipfile.ZIP_STORED:
                errors.append("mimetype must be stored without compression")
            elif archive.read("mimetype") != b"application/epub+zip":
                errors.append("mimetype payload is not application/epub+zip")
            if "META-INF/container.xml" not in member_set:
                return errors + ["missing META-INF/container.xml"]

            container = ET.fromstring(archive.read("META-INF/container.xml"))
            rootfiles = container.findall(".//container:rootfile", CONTAINER_NS)
            if len(rootfiles) != 1:
                errors.append(f"expected one OPF rootfile, found {len(rootfiles)}")
                return errors
            opf_path = rootfiles[0].attrib.get("full-path", "")
            if opf_path not in member_set:
                return errors + [f"container OPF is absent: {opf_path}"]

            opf = ET.fromstring(archive.read(opf_path))
            manifest = {
                item.attrib["id"]: item.attrib["href"]
                for item in opf.findall(".//opf:manifest/opf:item", OPF_NS)
                if "id" in item.attrib and "href" in item.attrib
            }
            spine = [item.attrib.get("idref", "") for item in opf.findall(".//opf:spine/opf:itemref", OPF_NS)]
            if not spine:
                errors.append("OPF spine is empty")
            for item_id in spine:
                if item_id not in manifest:
                    errors.append(f"spine item missing from manifest: {item_id}")
            for item_id, href in manifest.items():
                target = member_path(opf_path, href)
                if target not in member_set:
                    errors.append(f"manifest item {item_id} is absent: {target}")

            id_cache: dict[str, set[str]] = {}
            for member in (name for name in names if name.endswith((".xhtml", ".html"))):
                try:
                    document = ET.fromstring(archive.read(member))
                except ET.ParseError as error:
                    errors.append(f"invalid XHTML {member}: {error}")
                    continue
                id_cache[member] = {node.attrib["id"] for node in document.iter() if "id" in node.attrib}
                for node in document.iter():
                    for attr in ("href", "src"):
                        ref = node.attrib.get(attr)
                        if not ref or ref.startswith(("https:", "http:", "mailto:", "tel:", "data:")):
                            continue
                        target_ref, marker, fragment = ref.partition("#")
                        target = member if not target_ref else member_path(member, target_ref)
                        if target not in member_set:
                            errors.append(f"{member}: missing {attr} target {ref}")
                            continue
                        if marker and target.endswith((".xhtml", ".html")):
                            try:
                                target_ids = id_cache.setdefault(target, xhtml_ids(archive, target))
                            except ET.ParseError as error:
                                errors.append(f"invalid XHTML {target}: {error}")
                                continue
                            if fragment not in target_ids:
                                errors.append(f"{member}: missing fragment {ref}")
    except (OSError, zipfile.BadZipFile, ET.ParseError) as error:
        errors.append(str(error))
    return errors


all_errors: list[str] = []
for relative in EPUBS:
    epub = ROOT / relative
    if not epub.is_file():
        all_errors.append(f"missing EPUB: {relative}")
        continue
    errors = validate_epub(epub)
    print(f"EPUB {'PASS' if not errors else 'FAIL'}: {relative}")
    all_errors.extend(f"{relative}: {error}" for error in errors)

if all_errors:
    print("\n".join(all_errors), file=sys.stderr)
    raise SystemExit(1)
