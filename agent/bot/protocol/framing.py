import struct
import zlib
import json
import asyncio
from enum import IntEnum
from typing import Tuple, Optional, Dict, Any

MAGIC = b"MB"  # 2 bytes
PROTOCOL_VERSION = 1
HEADER_SIZE = 16  # 2 + 1 + 1 + 4 + 4 + 4

class MessageType(IntEnum):
    HELLO = 1
    WELCOME = 2
    OBSERVATION = 3
    ACTION = 4
    ACTION_RESULT = 5
    RESET = 6
    DEATH = 7
    PING = 8
    PONG = 9
    ERROR = 10
    SHUTDOWN = 11

def encode_frame(msg_type: MessageType, seq_id: int, payload_data: Dict[str, Any]) -> bytes:
    """
    Packs a frame with a 16-byte binary header:
      [Magic(2)][Version(1)][Type(1)][Seq(4)][Length(4)][CRC32(4)][Payload(Length)]
    """
    payload_bytes = json.dumps(payload_data, separators=(",", ":")).encode("utf-8")
    length = len(payload_bytes)
    crc = zlib.crc32(payload_bytes) & 0xFFFFFFFF
    header = struct.pack("!2sBBIII", MAGIC, PROTOCOL_VERSION, int(msg_type), seq_id, length, crc)
    return header + payload_bytes

def decode_header(header_bytes: bytes) -> Tuple[int, int, int, int]:
    """
    Decodes and validates the 16-byte header.
    Returns: (msg_type, seq_id, length, crc)
    """
    if len(header_bytes) != HEADER_SIZE:
        raise ValueError(f"Header must be {HEADER_SIZE} bytes, got {len(header_bytes)}")
    magic, version, msg_type, seq_id, length, crc = struct.unpack("!2sBBIII", header_bytes)
    if magic != MAGIC:
        raise ValueError(f"Invalid magic bytes: {magic}")
    if version != PROTOCOL_VERSION:
        raise ValueError(f"Unsupported protocol version: {version}")
    return msg_type, seq_id, length, crc

class StreamFramingReader:
    """Asynchronous reader parsing 16-byte framed messages over an asyncio StreamReader."""
    def __init__(self, reader: asyncio.StreamReader):
        self.reader = reader

    async def read_frame(self) -> Optional[Tuple[MessageType, int, Dict[str, Any]]]:
        try:
            header_bytes = await self.reader.readexactly(HEADER_SIZE)
        except (asyncio.IncompleteReadError, ConnectionResetError):
            return None

        msg_type_int, seq_id, length, expected_crc = decode_header(header_bytes)
        try:
            payload_bytes = await self.reader.readexactly(length)
        except (asyncio.IncompleteReadError, ConnectionResetError):
            return None

        actual_crc = zlib.crc32(payload_bytes) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise ValueError(f"CRC32 mismatch in frame seq {seq_id}: expected {expected_crc}, got {actual_crc}")

        payload = json.loads(payload_bytes.decode("utf-8"))
        try:
            msg_type = MessageType(msg_type_int)
        except ValueError:
            msg_type = MessageType.ERROR

        return msg_type, seq_id, payload
