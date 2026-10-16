"""Readers for ASE-style SQLite databases."""

from __future__ import annotations

import json
import sqlite3
import struct
from array import array
from pathlib import Path
from typing import Any

from atom_dataset_index.domain import DatasetBundle, DomainValidationError, StructureRecord


class DatasetReadError(RuntimeError):
    """Base exception raised for dataset read failures."""


class DatasetSchemaError(DatasetReadError):
    """Raised when required ASE schema columns are unavailable."""


class DatasetDecodeError(DatasetReadError):
    """Raised when ASE binary payloads cannot be decoded safely."""


class AseDbDatasetReader:
    """Read a subset of ASE DB fields into package domain objects."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)

    def read(self, dataset_name: str | None = None, limit: int | None = None) -> DatasetBundle:
        if not self.db_path.exists():
            raise FileNotFoundError(self.db_path)
        if limit is not None and limit <= 0:
            raise ValueError("limit must be greater than zero when provided.")

        try:
            with sqlite3.connect(self.db_path) as connection:
                connection.row_factory = sqlite3.Row
                self._ensure_required_schema(connection)
                metadata = self._read_metadata(connection)
                rows = self._read_rows(connection, limit=limit)
        except sqlite3.Error as error:
            raise DatasetReadError(f"Failed to read ASE DB: {self.db_path}") from error

        records = tuple(self._row_to_record(row) for row in rows)
        return DatasetBundle(
            dataset_name=dataset_name or self.db_path.stem,
            records=records,
            metadata=metadata,
        )

    def _ensure_required_schema(self, connection: sqlite3.Connection) -> None:
        if not self._table_exists(connection, "systems"):
            raise DatasetSchemaError("ASE database is missing required 'systems' table.")
        required_columns = {
            "id",
            "numbers",
            "positions",
            "cell",
            "pbc",
            "energy",
            "forces",
            "key_value_pairs",
        }
        cursor = connection.execute("PRAGMA table_info(systems)")
        columns = {str(row[1]) for row in cursor.fetchall()}
        missing = sorted(required_columns - columns)
        if missing:
            raise DatasetSchemaError(
                "ASE database systems table is missing required columns: "
                + ", ".join(missing)
            )

    @staticmethod
    def _table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
        cursor = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        return cursor.fetchone() is not None

    def _read_metadata(self, connection: sqlite3.Connection) -> dict[str, str | int | float | bool | None]:
        if not self._table_exists(connection, "information"):
            return {}
        info_columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(information)").fetchall()}
        key_column = "key" if "key" in info_columns else "name" if "name" in info_columns else None
        value_column = "value" if "value" in info_columns else None
        if key_column is None or value_column is None:
            return {}
        cursor = connection.execute(f"SELECT {key_column}, {value_column} FROM information")
        metadata: dict[str, str | int | float | bool | None] = {}
        for key, value in cursor.fetchall():
            normalized_key = str(key)
            if isinstance(value, (str, int, float, bool)) or value is None:
                metadata[normalized_key] = value
            else:
                metadata[normalized_key] = str(value)
        return metadata

    @staticmethod
    def _read_rows(connection: sqlite3.Connection, limit: int | None) -> list[sqlite3.Row]:
        columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(systems)").fetchall()}
        data_column = "data" if "data" in columns else "NULL AS data"
        query = (
            "SELECT id, numbers, positions, cell, pbc, energy, forces, key_value_pairs, "
            f"{data_column} "
            "FROM systems ORDER BY id"
        )
        params: tuple[int, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            params = (limit,)
        cursor = connection.execute(query, params)
        return list(cursor.fetchall())

    def _row_to_record(self, row: sqlite3.Row) -> StructureRecord:
        atomic_numbers = self._decode_int_array(row["numbers"])
        positions = self._decode_triplets(row["positions"])
        cell_rows = self._decode_triplets(row["cell"])
        cell = tuple(cell_rows[:3]) if len(cell_rows) >= 3 else None
        payload = self._decode_data_payload(row["data"])
        forces = self._decode_triplets(row["forces"])
        if not forces:
            forces = self._decode_triplets_from_payload(payload.get("forces"))
        energy = row["energy"]
        if energy is None:
            energy = self._decode_energy_from_payload(payload.get("energy"))
        pbc = self._decode_pbc(row["pbc"])
        metadata = self._decode_metadata(row["key_value_pairs"])
        if energy is None:
            raise DatasetSchemaError(
                f"Missing energy for sample_id={row['id']!r}; direct column and data payload are both empty."
            )
        if positions and not forces:
            raise DatasetSchemaError(
                f"Missing forces for sample_id={row['id']!r}; direct column and data payload are both empty."
            )
        try:
            return StructureRecord(
                sample_id=int(row["id"]),
                atomic_numbers=atomic_numbers,
                positions=positions,
                cell=cell,
                pbc=pbc,
                energy=float(energy) if energy is not None else None,
                forces=forces,
                metadata=metadata,
            )
        except (TypeError, ValueError, DomainValidationError) as error:
            raise DatasetSchemaError(
                f"Invalid row payload for sample_id={row['id']!r}."
            ) from error

    @staticmethod
    def _decode_int_array(blob: bytes | None) -> tuple[int, ...]:
        if not blob:
            return ()
        if not isinstance(blob, (bytes, bytearray, memoryview)):
            raise DatasetDecodeError("Expected binary integer array blob.")
        if len(blob) % array("i").itemsize != 0:
            raise DatasetDecodeError("Integer array payload has invalid byte length.")
        values = array("i")
        try:
            values.frombytes(bytes(blob))
        except ValueError as error:
            raise DatasetDecodeError("Failed to decode integer array payload.") from error
        return tuple(int(value) for value in values)

    @staticmethod
    def _decode_triplets(blob: bytes | None) -> tuple[tuple[float, float, float], ...]:
        if not blob:
            return ()
        if not isinstance(blob, (bytes, bytearray, memoryview)):
            raise DatasetDecodeError("Expected binary float array blob.")
        if len(blob) % array("d").itemsize != 0:
            raise DatasetDecodeError("Float array payload has invalid byte length.")
        values = array("d")
        try:
            values.frombytes(bytes(blob))
        except ValueError as error:
            raise DatasetDecodeError("Failed to decode float array payload.") from error
        numbers = list(values)
        if len(numbers) % 3 != 0:
            raise DatasetDecodeError("Float array payload cannot be reshaped into triplets.")
        return tuple(
            (numbers[index], numbers[index + 1], numbers[index + 2])
            for index in range(0, len(numbers), 3)
            if index + 2 < len(numbers)
        )

    @staticmethod
    def _decode_pbc(value: int | bytes | str | None) -> tuple[bool, bool, bool]:
        if value is None:
            return (False, False, False)
        bits: int
        if isinstance(value, bytes):
            bits = int.from_bytes(value, byteorder="little", signed=False)
        elif isinstance(value, str):
            bits = int(value)
        else:
            bits = int(value)
        if bits < 0:
            raise DatasetDecodeError("PBC flags must be non-negative.")
        return (bool(bits & 1), bool(bits & 2), bool(bits & 4))

    @staticmethod
    def _decode_metadata(
        raw_value: str | bytes | dict[str, object] | None
    ) -> dict[str, str | int | float | bool | None]:
        if not raw_value:
            return {}
        if isinstance(raw_value, bytes):
            raw_value = raw_value.decode("utf-8", errors="replace")
        if isinstance(raw_value, dict):
            return {
                str(key): AseDbDatasetReader._normalize_metadata_value(value)
                for key, value in raw_value.items()
            }
        try:
            data = json.loads(raw_value)
        except json.JSONDecodeError:
            return {"raw_key_value_pairs": raw_value}
        if isinstance(data, dict):
            return {
                str(key): AseDbDatasetReader._normalize_metadata_value(value)
                for key, value in data.items()
            }
        return {"raw_key_value_pairs": raw_value}

    @staticmethod
    def _normalize_metadata_value(value: object) -> str | int | float | bool | None:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return json.dumps(value, ensure_ascii=True, sort_keys=True)

    @staticmethod
    def _decode_data_payload(raw_value: bytes | memoryview | None) -> dict[str, Any]:
        if raw_value is None:
            return {}
        if isinstance(raw_value, memoryview):
            raw_value = raw_value.tobytes()
        if not isinstance(raw_value, (bytes, bytearray)):
            return {}
        try:
            from ase.db.core import bytes_to_object
        except ImportError:
            bytes_to_object = None
        if bytes_to_object is not None:
            try:
                payload = bytes_to_object(bytes(raw_value))
            except Exception:  # noqa: BLE001
                payload = None
            if isinstance(payload, dict):
                return payload
        try:
            return AseDbDatasetReader._decode_inline_ndarray_payload(bytes(raw_value))
        except Exception:  # noqa: BLE001
            return {}

    @staticmethod
    def _decode_inline_ndarray_payload(raw_value: bytes) -> dict[str, Any]:
        if len(raw_value) < 8:
            return {}
        header_offset = struct.unpack("<Q", raw_value[:8])[0]
        if header_offset >= len(raw_value):
            return {}
        header = json.loads(raw_value[header_offset:].decode("utf-8"))
        if not isinstance(header, dict):
            return {}
        return {
            key: AseDbDatasetReader._decode_inline_value(value, raw_value)
            for key, value in header.items()
        }

    @staticmethod
    def _decode_inline_value(value: Any, raw_value: bytes) -> Any:
        if isinstance(value, dict) and "__ndarray__" in value:
            descriptor = value["__ndarray__"]
            if not isinstance(descriptor, list) or len(descriptor) != 3:
                return value
            shape, dtype_name, offset = descriptor
            if not isinstance(shape, list) or not isinstance(dtype_name, str) or not isinstance(offset, int):
                return value
            return AseDbDatasetReader._read_ndarray(raw_value, tuple(int(item) for item in shape), dtype_name, offset)
        if isinstance(value, dict):
            return {
                str(key): AseDbDatasetReader._decode_inline_value(nested, raw_value)
                for key, nested in value.items()
            }
        if isinstance(value, list):
            return [AseDbDatasetReader._decode_inline_value(item, raw_value) for item in value]
        return value

    @staticmethod
    def _read_ndarray(
        raw_value: bytes,
        shape: tuple[int, ...],
        dtype_name: str,
        offset: int,
    ) -> Any:
        if dtype_name == "float64":
            format_code = "d"
        elif dtype_name == "float32":
            format_code = "f"
        elif dtype_name == "int32":
            format_code = "i"
        else:
            return []
        buffer = array(format_code)
        itemsize = buffer.itemsize
        count = 1
        for dimension in shape:
            count *= max(dimension, 1)
        end = offset + count * itemsize
        if offset < 0 or end > len(raw_value):
            return []
        buffer.frombytes(raw_value[offset:end])
        values = list(buffer)
        return AseDbDatasetReader._reshape(values, shape)

    @staticmethod
    def _reshape(values: list[Any], shape: tuple[int, ...]) -> Any:
        if not shape:
            return values[0] if values else None
        if len(shape) == 1:
            return values[: shape[0]]
        step = 1
        for dimension in shape[1:]:
            step *= dimension
        return [
            AseDbDatasetReader._reshape(values[index : index + step], shape[1:])
            for index in range(0, len(values), step)
        ]

    @staticmethod
    def _decode_energy_from_payload(value: Any) -> float | None:
        if value is None:
            return None
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            if not value:
                return None
            value = value[0]
            if isinstance(value, (list, tuple)):
                if not value:
                    return None
                value = value[0]
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _decode_triplets_from_payload(value: Any) -> tuple[tuple[float, float, float], ...]:
        if value is None:
            return ()
        if hasattr(value, "tolist"):
            value = value.tolist()
        if not isinstance(value, (list, tuple)):
            return ()
        if value and isinstance(value[0], (list, tuple)):
            vectors: list[tuple[float, float, float]] = []
            for row in value:
                if not isinstance(row, (list, tuple)) or len(row) < 3:
                    continue
                vectors.append((float(row[0]), float(row[1]), float(row[2])))
            return tuple(vectors)
        if len(value) % 3 != 0:
            return ()
        vectors = []
        for index in range(0, len(value), 3):
            vectors.append((float(value[index]), float(value[index + 1]), float(value[index + 2])))
        return tuple(vectors)
