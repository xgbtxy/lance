"""Probe whether a writer-only manifest flag protects a v3 code FTS index.

Use separate CPython 3.12 environments for official pylance==14.0.2 and
pylance==8.0.1 wheels. See RESULT.txt for the runnable command sequence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct

import lance
import pyarrow as pa

WRITER_FLAGS = 10
READER_FLAGS = 9
PROBE_BIT = 1 << 20
BASE_ROWS = [
    {"id": 0, "code": "def getUserName(): pass"},
    {"id": 1, "code": "class HTTPServer: pass"},
]


def read_varint(data, offset):
    value = shift = 0
    while True:
        byte = data[offset]
        offset += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, offset
        shift += 7
        if shift >= 70:
            raise ValueError("invalid protobuf varint")


def put_varint(value):
    result = bytearray()
    while value >= 128:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def fields(data):
    found = {}
    offset = 0
    while offset < len(data):
        key, offset = read_varint(data, offset)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, offset = read_varint(data, offset)
            found.setdefault(number, []).append(value)
        elif wire == 1:
            offset += 8
        elif wire == 2:
            size, offset = read_varint(data, offset)
            offset += size
        elif wire == 5:
            offset += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        if offset > len(data):
            raise ValueError("protobuf field exceeds message length")
    return found


def manifest(path):
    data = path.read_bytes()
    if data[-4:] != b"LANC":
        raise ValueError(f"bad manifest footer: {path}")
    start = struct.unpack_from("<q", data, len(data) - 16)[0]
    length = struct.unpack_from("<I", data, start)[0]
    body_start = start + 4
    body_end = body_start + length
    if body_end != len(data) - 16:
        raise ValueError(f"manifest length mismatch: {path}")
    return data, start, data[body_start:body_end], data[body_end:]


def add_varint_field(body, number, value):
    output = bytearray()
    offset = 0
    seen = False
    while offset < len(body):
        start = offset
        key, offset = read_varint(body, offset)
        field, wire = key >> 3, key & 7
        if wire == 0:
            old, offset = read_varint(body, offset)
            if field == number:
                if seen:
                    raise ValueError(f"duplicate field {number}")
                seen = True
                output.extend(body[start : start + len(put_varint(key))])
                output.extend(put_varint(old | value))
            else:
                output.extend(body[start:offset])
        elif wire == 1:
            offset += 8
            output.extend(body[start:offset])
        elif wire == 2:
            size, payload = read_varint(body, offset)
            offset = payload + size
            output.extend(body[start:offset])
        elif wire == 5:
            offset += 4
            output.extend(body[start:offset])
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
    if not seen:
        output.extend(put_varint((number << 3) | 0))
        output.extend(put_varint(value))
    return bytes(output)


def versioned_manifests(path):
    directory = path / "_versions"
    hint = json.loads((directory / "latest_version_hint.json").read_text())["version"]
    candidates = []
    for item in directory.glob("*.manifest"):
        _, _, body, _ = manifest(item)
        version = fields(body).get(3, [None])[0]
        if version is not None:
            candidates.append((version, item, body))
    version, item, body = max(candidates, key=lambda row: row[0])
    if version != hint:
        raise ValueError(f"Manifest.version {version} != latest hint {hint}")
    return version, item, body


def hashes(directory):
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.rglob("*")
        if path.is_file()
    }


def create(path):
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = lance.write_dataset(pa.Table.from_pylist(BASE_ROWS), str(path))
    dataset.create_scalar_index(
        "code", "INVERTED", analyzer="code", split_identifiers=True
    )
    check = lance.dataset(str(path))
    print(json.dumps({
        "pylance": lance.__version__,
        "dataset_version": check.version,
        "indices": [item.name for item in check.describe_indices()],
        "rows": check.to_table().to_pylist(),
    }))


def flag(source, destination):
    source, destination = source.resolve(), destination.resolve()
    if not source.is_dir() or destination.exists() or source == destination:
        raise ValueError("source must exist and destination must be new and distinct")
    source_hashes = hashes(source)
    shutil.copytree(source, destination)
    version, path, body = versioned_manifests(destination)
    before = fields(body)
    if before.get(READER_FLAGS, [0])[0] != 0:
        raise ValueError("probe assumes reader_feature_flags is zero")
    new_body = add_varint_field(
        body, WRITER_FLAGS, PROBE_BIT
    )
    original, start, _, footer = manifest(path)
    path.write_bytes(original[:start] + struct.pack("<I", len(new_body)) + new_body + footer)
    after = fields(manifest(path)[2])
    if after.get(3) != [version] or after.get(READER_FLAGS, [0])[0] != 0:
        raise ValueError("manifest version or reader flags changed")
    expected_writer = before.get(WRITER_FLAGS, [0])[0] | PROBE_BIT
    if after.get(WRITER_FLAGS) != [expected_writer]:
        raise ValueError("writer flag did not round-trip")
    after_hashes = hashes(destination)
    changed = [name for name in source_hashes if source_hashes[name] != after_hashes[name]]
    if changed != [str(path.relative_to(destination))] or source_hashes != hashes(source):
        raise ValueError(f"unexpected fixture changes: {changed}")
    print(json.dumps({
        "pylance": lance.__version__,
        "manifest_version": version,
        "reader_feature_flags": after.get(READER_FLAGS, [0]),
        "writer_feature_flags": after.get(WRITER_FLAGS),
        "only_changed_file": changed[0],
        "source_unchanged": True,
    }))


def read_dataset(path):
    dataset = lance.dataset(str(path))
    return dataset, {
        "version": dataset.version,
        "rows": dataset.to_table().to_pylist(),
        "indices": [item.name for item in dataset.describe_indices()],
    }


def read(path):
    dataset, state = read_dataset(path)
    query = dataset.to_table(full_text_query="user")
    print(json.dumps({
        "pylance": lance.__version__,
        "dataset": state,
        "fts_rows": query.num_rows,
        "fts_code": query["code"].to_pylist(),
    }))


def probe(path, method):
    path = path.resolve()
    before_files = hashes(path)
    _, before = read_dataset(path)
    incoming = pa.Table.from_pylist([{"id": 2, "code": "fn fetchUserProfile() {}"}])
    error = None
    try:
        dataset = lance.dataset(str(path))
        if method == "insert":
            dataset.insert(incoming)
        else:
            lance.write_dataset(incoming, dataset, mode="append")
    except Exception as exc:
        error = str(exc).splitlines()[0]
    _, after = read_dataset(path)
    if error is None or "cannot be written" not in error.lower():
        raise AssertionError(f"writer did not issue the expected refusal: {error}")
    if before != after or before_files != hashes(path):
        raise AssertionError("dataset changed after refused write")
    result = {
        "pylance": lance.__version__,
        "method": method,
        "write_refused": True,
        "error": error,
        "before_after": before,
        "files_unchanged": True,
    }
    if lance.__version__.startswith("14."):
        dataset = lance.dataset(str(path))
        query = dataset.to_table(full_text_query="user")
        if query.num_rows != 1 or query["code"].to_pylist() != [BASE_ROWS[0]["code"]]:
            raise AssertionError("current reader lost the v3 FTS result")
        result.update(fts_rows=query.num_rows, fts_code=query["code"].to_pylist())
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create_cmd = commands.add_parser("create")
    create_cmd.add_argument("--dataset", type=Path, required=True)
    flag_cmd = commands.add_parser("flag")
    flag_cmd.add_argument("--source", type=Path, required=True)
    flag_cmd.add_argument("--dataset", type=Path, required=True)
    for name in ("read", "probe"):
        command = commands.add_parser(name)
        command.add_argument("--dataset", type=Path, required=True)
        if name == "probe":
            command.add_argument("--method", choices=("insert", "append"), required=True)
    args = parser.parse_args()
    if args.command == "create":
        create(args.dataset)
    elif args.command == "flag":
        flag(args.source, args.dataset)
    elif args.command == "read":
        read(args.dataset)
    else:
        probe(args.dataset, args.method)


if __name__ == "__main__":
    main()
