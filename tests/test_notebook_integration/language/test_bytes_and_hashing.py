"""bytes, encodings, base64, hashlib and hmac across cells."""

import textwrap

import pytest


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


# bytes encoding and decoding patterns with caching.
# Tests str.encode, bytes.decode, hex conversion, and edit propagation.
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


# Interaction test: bytes and bytearray encoding operations.
# Tests bytes/bytearray construction, hex conversion,
# encoding/decoding, and cross-cell binary pipelines.
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


# String encoding and bytes conversion interaction tests.
#
# Tests editing string/bytes conversions, encoding schemes,
# and byte manipulation patterns.
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
class TestHashlibDigest:
    """hashlib hashing and digest comparison."""

    def test_sha256(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import hashlib\ntext = 'Hello, World!'",
                "h = hashlib.sha256(text.encode()).hexdigest()\nprint(f'hash={h[:16]}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "hash=dffd6021bb2bd5b0" in nb_runner.get_output(2)

    def test_hash_edit_input(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import hashlib\nmsg = 'test'",
                "h = hashlib.md5(msg.encode()).hexdigest()\nprint(f'md5={h}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "md5=098f6bcd4621d373cade4e832627b4f6" in nb_runner.get_output(2)
        # Edit
        nb_runner.set_cell_source(1, "import hashlib\nmsg = 'hello'")
        nb_runner.run_all()
        assert "md5=5d41402abc4b2a76b9719d911017c592" in nb_runner.get_output(2)

    def test_hash_compare(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import hashlib\ndef hash_str(s):\n    return hashlib.sha1(s.encode()).hexdigest()[:8]",
                "h1 = hash_str('abc')\nh2 = hash_str('abc')\nh3 = hash_str('xyz')\nmatch = h1 == h2\ndiff = h1 != h3\nprint(f'match={match} diff={diff}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "match=True diff=True" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestHashlibSha256Md5:
    """hashlib sha256 md5 hexdigest computation."""

    def test_sha256(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import hashlib",
                "data = b'hello world'\nh = hashlib.sha256(data).hexdigest()\nprint(f'sha256={h[:16]}... len={len(h)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "sha256=b94d27b9934d3e08" in out
        assert "len=64" in out

    def test_md5(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import hashlib",
                "data = b'test'\nh = hashlib.md5(data).hexdigest()\nprint(f'md5={h} len={len(h)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "md5=098f6bcd4621d373cade4e832627b4f6" in out
        assert "len=32" in out

    def test_hash_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import hashlib",
                "h = hashlib.sha256(b'abc').hexdigest()[:8]\nprint(f'h={h}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "h=ba7816bf" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "h = hashlib.sha256(b'xyz').hexdigest()[:8]\nprint(f'h={h}')")
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "h=" in out2
        assert "ba7816bf" not in out2  # different from abc


# Interaction test: hashlib HMAC for message authentication.
# Tests hmac.new with hashlib digests, compare_digest for timing-safe
# comparison, and cross-cell HMAC verification pipelines.
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestHmacAuth:
    """Test HMAC message authentication across cells."""

    def test_hmac_ops(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: create HMAC
                "import hmac\nimport hashlib\nkey = b'secret-key'\nmessage = b'important data'\nmac = hmac.new(key, message, hashlib.sha256).hexdigest()\nprint(f'mac_len={len(mac)}')\nprint(f'mac_prefix={mac[:8]}')",
                # Cell 2: verify HMAC
                "mac2 = hmac.new(key, message, hashlib.sha256).hexdigest()\nis_valid = hmac.compare_digest(mac, mac2)\nprint(f'valid={is_valid}')",
                # Cell 3: different message = different HMAC
                "bad_mac = hmac.new(key, b'tampered data', hashlib.sha256).hexdigest()\nis_tampered = not hmac.compare_digest(mac, bad_mac)\nprint(f'tampered={is_tampered}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "mac_len=64" in out1
        out2 = nb_runner.get_output(2)
        assert "valid=True" in out2
        out3 = nb_runner.get_output(3)
        assert "tampered=True" in out3

    def test_hmac_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import hmac, hashlib\nkey = b'key1'\nmsg = b'hello'\ntag = hmac.new(key, msg, hashlib.sha256).hexdigest()\nprint(f'tag={tag[:12]}')",
                "tag_short = tag[:8]\nprint(f'short={tag_short}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out2a = nb_runner.get_output(2)
        assert "short=" in out2a

        # Edit key
        nb_runner.set_cell_source(
            1,
            "import hmac, hashlib\nkey = b'key2'\nmsg = b'hello'\ntag = hmac.new(key, msg, hashlib.sha256).hexdigest()\nprint(f'tag={tag[:12]}')",
        )
        nb_runner.run_cells([1, 2])
        out2b = nb_runner.get_output(2)
        assert "short=" in out2b

    def test_hmac_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import hmac, hashlib\nresult = hmac.new(b'k', b'data', hashlib.md5).hexdigest()\nprint(f'result_len={len(result)}')",
                "is_32 = len(result) == 32\nprint(f'is_32={is_32}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result_len=32" in nb_runner.get_output(1)
        assert "is_32=True" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "is_32=True" in nb_runner.get_output(2)


# Interaction test: io.StringIO and io.BytesIO in-memory streams.
# Tests reading/writing to in-memory buffers, seek/tell operations,
# cross-cell stream sharing, and cache invalidation on content changes.
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


# Hashlib, secrets & crypto patterns — cash caching with hashing/security.
@pytest.mark.stress
class TestHashlibPatterns:
    """Test hashlib patterns across cells."""

    def test_hash_file_content(self, nb_runner, tmp_path):
        """Hash file content across cells."""
        test_file = tmp_path / "hashdata" / "sample.txt"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("Test content for hashing", encoding="utf-8")
        fpath = str(test_file).replace("\\", "/")

        nb_runner.create_notebook(
            [
                textwrap.dedent(f"""\
                import hashlib

                with open('{fpath}', 'rb') as f:
                    content = f.read()
                file_hash = hashlib.sha256(content).hexdigest()
                print(f"hash={{file_hash[:16]}}")
            """),
                textwrap.dedent("""\
                print(f"content_size={len(content)}")
                print(f"hash_type=sha256")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "hash=" in nb_runner.get_output(1)
        assert "content_size=24" in nb_runner.get_output(2)


@pytest.mark.stress
class TestSecretsPatterns:
    """Test secrets module patterns."""

    def test_hash_change_propagation(self, nb_runner):
        """Hash result propagates when input changes."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                import hashlib
                data = "version1"
                digest = hashlib.sha256(data.encode()).hexdigest()
                print(f"digest={digest[:16]}")
            """),
                textwrap.dedent("""\
                short = digest[:8]
                print(f"short={short}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        first_short = nb_runner.get_output(2).split("short=")[1].strip()

        # Change data
        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            import hashlib
            data = "version2"
            digest = hashlib.sha256(data.encode()).hexdigest()
            print(f"digest={digest[:16]}")
        """),
        )
        nb_runner.run_cells([1, 2])
        second_short = nb_runner.get_output(2).split("short=")[1].strip()
        assert first_short != second_short, "Hash should change when input changes"
