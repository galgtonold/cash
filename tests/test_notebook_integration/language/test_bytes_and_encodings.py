"""bytes, str encoding and decoding, base64 and in-memory streams across cells."""

import pytest


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestBytesEncode:
    """Test bytes/encoding operation caching."""

    def test_encode_decode_roundtrip(self, nb_runner):
        """Encode string to bytes, decode back, verify caching."""
        nb_runner.create_notebook(
            [
                "text = 'Hello, World!'",
                "encoded = text.encode('utf-8')",
                "decoded = encoded.decode('utf-8')\nprint(f'match={decoded == text}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "match=True" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "match=True" in out2

    def test_hex_conversion_edit(self, nb_runner):
        """Hex string conversion with edit."""
        nb_runner.create_notebook(
            [
                "data = b'\\x48\\x65\\x6c\\x6c\\x6f'",
                "hex_str = data.hex()\ntext = data.decode('ascii')",
                "print(f'hex={hex_str} text={text}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "hex=48656c6c6f" in out
        assert "text=Hello" in out

        nb_runner.set_cell_source(1, "data = b'\\x57\\x6f\\x72\\x6c\\x64'")
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "text=World" in out2

    def test_base64_encode(self, nb_runner):
        """Base64 encoding with caching."""
        nb_runner.create_notebook(
            [
                "import base64",
                "msg = 'test data'",
                "enc = base64.b64encode(msg.encode()).decode()\ndec = base64.b64decode(enc).decode()",
                "print(f'enc={enc} match={dec == msg}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "match=True" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "match=True" in out2


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestBytesEncodingOps:
    """Test bytes and bytearray encoding across cells."""

    def test_bytes_encoding(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: create bytes
                "text = 'Hello, World!'\nencoded = text.encode('utf-8')\nhex_str = encoded.hex()\nprint(f'length={len(encoded)}')\nprint(f'hex={hex_str}')",
                # Cell 2: decode and bytearray
                "decoded = encoded.decode('utf-8')\nba = bytearray(encoded)\nba[0] = ord('h')  # lowercase\nmodified = ba.decode('utf-8')\nprint(f'decoded={decoded}')\nprint(f'modified={modified}')",
                # Cell 3: from hex roundtrip
                "restored = bytes.fromhex(hex_str)\nprint(f'restored={restored.decode(\"utf-8\")}')\nprint(f'matches={restored == encoded}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "length=13" in out1
        out2 = nb_runner.get_output(2)
        assert "decoded=Hello, World!" in out2
        assert "modified=hello, World!" in out2
        out3 = nb_runner.get_output(3)
        assert "restored=Hello, World!" in out3
        assert "matches=True" in out3

    def test_bytes_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "data = b'\\x01\\x02\\x03\\x04'\nprint(f'hex={data.hex()}')",
                "total = sum(data)\nprint(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=10" in nb_runner.get_output(2)

        # Edit to different bytes
        nb_runner.set_cell_source(1, "data = b'\\x0a\\x14\\x1e\\x28'\nprint(f'hex={data.hex()}')")
        nb_runner.run_cells([1, 2])
        # 0x0a=10, 0x14=20, 0x1e=30, 0x28=40 => 100
        assert "total=100" in nb_runner.get_output(2)

    def test_bytes_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "msg = 'Python'\nencoded = msg.encode('ascii')\nprint(f'bytes={list(encoded)}')",
                "upper = bytes([b - 32 if 97 <= b <= 122 else b for b in encoded])\nprint(f'upper={upper.decode(\"ascii\")}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "upper=PYTHON" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "upper=PYTHON" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestEncodingEdits:
    """Editing encoding/decoding operations."""

    def test_edit_encoding_scheme(self, nb_runner):
        """Edit the encoding scheme."""
        nb_runner.create_notebook(
            [
                "text = 'Hello'  # encoding source",
                "encoded = text.encode('utf-8')\nprint(f'bytes = {encoded}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "bytes = b'Hello'" in nb_runner.get_output(2)

        # Change to ascii
        nb_runner.set_cell_source(2, "encoded = text.encode('ascii')\nprint(f'bytes = {encoded}')")
        nb_runner.run_all()
        assert "bytes = b'Hello'" in nb_runner.get_output(2)

    def test_edit_bytes_to_hex(self, nb_runner):
        """Edit bytes to hex conversion."""
        nb_runner.create_notebook(
            [
                "data = b'\\x48\\x65\\x6c\\x6c\\x6f'  # bytes hex source",
                "hex_str = data.hex()\nprint(f'hex = {hex_str}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "hex = 48656c6c6f" in nb_runner.get_output(2)

        # Change data
        nb_runner.set_cell_source(1, "data = b'\\x41\\x42\\x43'  # bytes hex source v2")
        nb_runner.run_all()
        assert "hex = 414243" in nb_runner.get_output(2)

    def test_edit_base64_roundtrip(self, nb_runner):
        """Edit base64 encode/decode."""
        nb_runner.create_notebook(
            [
                "import base64",
                "original = 'Hello World'  # base64 source",
                "encoded = base64.b64encode(original.encode()).decode()\nprint(f'encoded = {encoded}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "encoded = SGVsbG8gV29ybGQ=" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "original = 'Python'  # base64 source v2")
        nb_runner.run_all()
        assert "encoded = UHl0aG9u" in nb_runner.get_output(3)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestStringBytesEncode:
    """string encode/decode, base64, and bytes operations."""

    def test_encode_decode(self, nb_runner):
        nb_runner.create_notebook(
            [
                "text = 'Hello, World!'",
                "encoded = text.encode('utf-8')\ndecoded = encoded.decode('utf-8')\nhex_str = encoded.hex()\nprint(f'decoded={decoded} hex={hex_str[:10]}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "decoded=Hello, World!" in out

    def test_base64_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import base64\nmessage = 'Hello'",
                "encoded = base64.b64encode(message.encode()).decode()\nprint(f'encoded={encoded}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "encoded=SGVsbG8=" in nb_runner.get_output(2)
        # Edit
        nb_runner.set_cell_source(1, "import base64\nmessage = 'World'")
        nb_runner.run_all()
        assert "encoded=V29ybGQ=" in nb_runner.get_output(2)

    def test_bytes_operations(self, nb_runner):
        nb_runner.create_notebook(
            [
                "data = bytes([72, 101, 108, 108, 111])",
                "text = data.decode('ascii')\nlength = len(data)\nprint(f'text={text} length={length}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "text=Hello length=5" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestStringEncodeDecodeBytes:
    """string encode decode and bytes operations."""

    def test_encode_decode_utf8(self, nb_runner):
        nb_runner.create_notebook(
            [
                "text = 'Hello \\u00e9\\u00e8 \\u4e16\\u754c'",
                "encoded = text.encode('utf-8')\nsize = len(encoded)\ndecoded = encoded.decode('utf-8')\nprint(f'size={size} match={text == decoded}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "match=True" in out

    def test_bytes_operations(self, nb_runner):
        nb_runner.create_notebook(
            [
                "data = b'Hello World'",
                "upper = data.upper()\nlower = data.lower()\nfound = data.find(b'World')\nprint(f'upper={upper} lower={lower} found={found}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "upper=b'HELLO WORLD'" in out
        assert "lower=b'hello world'" in out
        assert "found=6" in out

    def test_encode_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "text = 'abc'",
                "b = text.encode('ascii')\nprint(f'len={len(b)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "len=3" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "text = 'abcdef'")
        nb_runner.run_all()
        assert "len=6" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestStringEncodingBytes:
    """string encoding and byte operations."""

    def test_encode_decode(self, nb_runner):
        nb_runner.create_notebook(
            [
                "text = 'hello world'",
                "encoded = text.encode('utf-8')\ndecoded = encoded.decode('utf-8')\nhex_repr = encoded.hex()\nprint(f'decoded={decoded} hex={hex_repr}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "decoded=hello world" in nb_runner.get_output(2)
        assert "hex=68656c6c6f20776f726c64" in nb_runner.get_output(2)

    def test_bytes_from_hex(self, nb_runner):
        nb_runner.create_notebook(
            [
                "hex_str = '48454c4c4f'",
                "data = bytes.fromhex(hex_str)\ntext = data.decode('ascii')\nprint(f'text={text} len={len(data)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "text=HELLO" in nb_runner.get_output(2)
        assert "len=5" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestBase64HexConversions:
    """base64 encode decode and hex conversions."""

    def test_base64_roundtrip(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import base64",
                "text = 'Hello, World!'\nencoded = base64.b64encode(text.encode()).decode()\ndecoded = base64.b64decode(encoded).decode()\nprint(f'encoded={encoded} decoded={decoded}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "encoded=SGVsbG8sIFdvcmxkIQ==" in out
        assert "decoded=Hello, World!" in out

    def test_hex_conversion(self, nb_runner):
        nb_runner.create_notebook(
            [
                "pass  # setup",
                "data = bytes([0, 127, 255, 16])\nhex_str = data.hex()\nback = bytes.fromhex(hex_str)\nprint(f'hex={hex_str} match={data == back}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "hex=007fff10" in out
        assert "match=True" in out

    def test_base64_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import base64",
                "msg = 'abc'\nenc = base64.b64encode(msg.encode()).decode()\nprint(f'enc={enc}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "enc=YWJj" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "msg = 'xyz'\nenc = base64.b64encode(msg.encode()).decode()\nprint(f'enc={enc}')")
        nb_runner.run_all()
        assert "enc=eHl6" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestIoStringBytesIO:
    """Test io module in-memory streams across cells."""

    def test_stringio_ops(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: write to StringIO
                "import io\nbuf = io.StringIO()\nbuf.write('Hello ')\nbuf.write('World')\ncontent = buf.getvalue()\nprint(f'content={content}')\nprint(f'length={len(content)}')",
                # Cell 2: read from StringIO
                "reader = io.StringIO(content)\nfirst_word = reader.read(5)\nprint(f'first={first_word}')\nrest = reader.read()\nprint(f'rest={rest}')",
                # Cell 3: BytesIO
                "bbuf = io.BytesIO(b'binary data')\nbbuf.seek(7)\nchunk = bbuf.read()\nprint(f'chunk={chunk}')\nprint(f'chunk_str={chunk.decode()}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "content=Hello World" in out1
        assert "length=11" in out1
        out2 = nb_runner.get_output(2)
        assert "first=Hello" in out2
        assert "rest= World" in out2
        out3 = nb_runner.get_output(3)
        assert "chunk_str=data" in out3

    def test_stringio_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import io\nbuf = io.StringIO()\nfor i in range(3):\n    buf.write(f'line{i}\\n')\ntext = buf.getvalue()\nline_count = text.strip().count('\\n') + 1\nprint(f'lines={line_count}')",
                "first_line = text.split('\\n')[0]\nprint(f'first={first_line}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "lines=3" in nb_runner.get_output(1)
        assert "first=line0" in nb_runner.get_output(2)

        # Edit range
        nb_runner.set_cell_source(
            1,
            "import io\nbuf = io.StringIO()\nfor i in range(5):\n    buf.write(f'line{i}\\n')\ntext = buf.getvalue()\nline_count = text.strip().count('\\n') + 1\nprint(f'lines={line_count}')",
        )
        nb_runner.run_cells([1, 2])
        assert "lines=5" in nb_runner.get_output(1)
        assert "first=line0" in nb_runner.get_output(2)

    def test_stringio_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import io\ncsv_data = 'a,b,c\\n1,2,3\\n4,5,6'\nreader = io.StringIO(csv_data)\nheader = reader.readline().strip()\nprint(f'header={header}')",
                "cols = header.split(',')\ncol_count = len(cols)\nprint(f'col_count={col_count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "header=a,b,c" in nb_runner.get_output(1)
        assert "col_count=3" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "col_count=3" in nb_runner.get_output(2)
